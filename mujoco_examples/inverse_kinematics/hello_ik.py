"""
This script uses the MuJoCo physics engine to control a UR5e robot arm to draw the word "Hello" in a cursive style.
The code is adapted from Kevin Zakka's MJCTRL codebase:
https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.interpolate import interp1d

integration_dt: float = 1.0
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.002

max_angvel = 1.0  # Limit velocity for smoother writing
word_scale = 0.15

drawing_x = 0.6  # X position of the vertical drawing plane
pen_out_x = drawing_x - 0.05  # X position when pen is lifted away from surface


def generate_letter_paths():
    """
    Generate parametric paths for cursive letters.
    Returns a dictionary with letter -> list of points.
    """
    letter_paths = {}
    
    # Define cursive letter paths using parametric points
    t = np.linspace(0, 1, 100)
    
    # Letter 'h'
    h_y = 0.3 * np.ones_like(t)
    h_z = 0.3 * np.sin(np.pi * t)
    # Add the hook at the top
    h_y[70:] = 0.3 + 0.15 * (t[70:] - t[70]) / (t[-1] - t[70])
    h_z[70:] = 0.3 - 0.15 * (t[70:] - t[70]) / (t[-1] - t[70])
    letter_paths['h'] = np.column_stack((h_y, h_z))
    
    # Letter 'e'
    theta = 2 * np.pi * t
    e_y = 0.15 * np.cos(theta) + 0.3
    e_z = 0.15 * np.sin(theta) + 0
    # Add entry and exit strokes
    e_y[:10] = np.linspace(0.15, 0.3 + 0.15, 10)
    e_z[:10] = np.linspace(0, 0, 10)
    letter_paths['e'] = np.column_stack((e_y, e_z))
    
    # Letter 'l'
    l_y = 0.1 * np.ones_like(t)
    l_z = 0.4 * t - 0.2
    # Add a small curve at the top
    l_y[80:] = 0.1 + 0.05 * (t[80:] - t[80]) / (t[-1] - t[80])
    letter_paths['l'] = np.column_stack((l_y, l_z))
    
    # Letter 'o'
    theta = 2 * np.pi * t
    o_y = 0.15 * np.cos(theta) + 0.3
    o_z = 0.15 * np.sin(theta) + 0
    letter_paths['o'] = np.column_stack((o_y, o_z))
    
    return letter_paths


def generate_word_path(word, letter_paths, spacing=0.4):
    """
    Generate a path for the entire word by combining letter paths.
    
    Args:
        word: String to generate path for
        letter_paths: Dictionary of letter paths
        spacing: Horizontal spacing between letters
    
    Returns:
        List of (x, y, z) points, with x indicating pen in/out
    """
    word_path = []
    current_y_offset = 0
    
    for i, letter in enumerate(word.lower()):
        if letter in letter_paths:
            letter_path = letter_paths[letter].copy()
            
            letter_path[:, 0] += current_y_offset
            
            first_point = letter_path[0].copy()
            word_path.append((pen_out_x, first_point[0], first_point[1]))
            
            for point in letter_path:
                word_path.append((drawing_x, point[0], point[1]))
            
            current_y_offset += spacing
            
            if i < len(word) - 1:
                last_point = letter_path[-1].copy()
                word_path.append((pen_out_x, last_point[0], last_point[1]))
        else:
            # For unsupported characters (spaces, etc.), just move horizontally
            current_y_offset += spacing / 2
    
    return np.array(word_path)


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    model = mujoco.MjModel.from_xml_path("../universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    # End-effector site we wish to control
    site_id = model.site("attachment_site").id

    # Name of bodies we wish to apply gravity compensation to
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    # Get the dof and actuator ids for the joints we wish to control
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration
    key_id = model.key("home").id

    # Mocap body we will control with our trajectory
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Generate the word path
    word_to_draw = "hello"  # Can be changed to any word with the supported letters
    letter_paths = generate_letter_paths()
    word_path = generate_word_path(word_to_draw, letter_paths)
    
    # Scale and center the path
    word_path[:, 1:3] = word_path[:, 1:3] * word_scale
    
    # Center the word vertically in the workspace
    word_path[:, 2] += 0.5 - np.mean(word_path[:, 2])
    
    # Create time parameter for the path
    # We want to move slower when the pen is on the surface and faster when the pen is out
    path_length = len(word_path)
    time_per_point = []
    for i in range(1, path_length):
        # Slower speed for drawing, faster for pen out movements
        if word_path[i, 0] == drawing_x:
            time_per_point.append(0.05)  # Slower when drawing
        else:
            time_per_point.append(0.01)  # Faster when pen is out
    
    # Cumulative time for each point
    cum_time = np.cumsum([0] + time_per_point)
    total_time = cum_time[-1]
    
    # Create interpolation functions for each coordinate
    interp_x = interp1d(cum_time, word_path[:, 0], bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(cum_time, word_path[:, 1], bounds_error=False, fill_value="extrapolate")
    interp_z = interp1d(cum_time, word_path[:, 2], bounds_error=False, fill_value="extrapolate")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        # Set orientation for the pen - pointing towards the vertical surface
        # This rotates the end-effector to point along the x-axis
        forward_quat = np.array([0.7071068, 0, 0.7071068, 0])  # Pointing forward (x-axis)
        data.mocap_quat[mocap_id] = forward_quat
        
        while viewer.is_running():
            step_start = time.time()
            
            # Get current time in the drawing cycle
            elapsed = (data.time % (total_time * 1.5))  # Add pause between cycles
            drawing_active = elapsed < total_time
            
            if drawing_active:
                # Get the target position from the interpolated path
                target_x = float(interp_x(elapsed))
                target_y = float(interp_y(elapsed))
                target_z = float(interp_z(elapsed))
                
                # Set the target position
                data.mocap_pos[mocap_id] = np.array([target_x, target_y, target_z])
            else:
                # Return to starting position during pause
                start_x = float(interp_x(0))
                start_y = float(interp_y(0))
                start_z = float(interp_z(0))
                data.mocap_pos[mocap_id] = np.array([pen_out_x, start_y, start_z])

            # Position error
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()
            
            # Control simulation speed
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()