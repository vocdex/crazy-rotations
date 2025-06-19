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
from traj_visualizer import RealTimeTrailVisualizer

integration_dt: float = 1.0
damping: float = 1e-3
gravity_compensation: bool = True
dt: float = 0.002

max_angvel = 1.0  # Limit velocity for smoother writing
word_scale = 0.2

table_height = 0.30
drawing_z = table_height + 0.02  
pen_out_z = table_height + 0.08  

radius = 0.05      # Smaller radius
center_x = 0.6     
center_y = 0.0
frequency = 0.2
def generate_letter_paths():
    """
    Generate parametric paths for cursive letters.
    Returns a dictionary with letter -> list of points.
    """
    letter_paths = {}
    
    # Define cursive letter paths using parametric points
    t = np.linspace(0, 1, 100)
    
    # Letter 'h'
    h_x = 0.3 * np.ones_like(t)
    h_y = 0.3 * np.sin(np.pi * t)
    # Add the hook at the top
    h_x[70:] = 0.3 + 0.15 * (t[70:] - t[70]) / (t[-1] - t[70])
    h_y[70:] = 0.3 - 0.15 * (t[70:] - t[70]) / (t[-1] - t[70])
    letter_paths['h'] = np.column_stack((h_x, h_y))
    
    # Letter 'e'
    theta = 2 * np.pi * t
    e_x = 0.15 * np.cos(theta) + 0.3
    e_y = 0.15 * np.sin(theta) + 0
    # Add entry and exit strokes
    e_x[:10] = np.linspace(0.15, 0.3 + 0.15, 10)
    e_y[:10] = np.linspace(0, 0, 10)
    letter_paths['e'] = np.column_stack((e_x, e_y))
    
    # Letter 'l'
    l_x = 0.1 * np.ones_like(t)
    l_y = 0.4 * t - 0.2
    letter_paths['l'] = np.column_stack((l_x, l_y))
    
    # Letter 'o'
    theta = 2 * np.pi * t
    o_x = 0.15 * np.cos(theta) + 0.3
    o_y = 0.15 * np.sin(theta) + 0
    letter_paths['o'] = np.column_stack((o_x, o_y))
    
    return letter_paths



def generate_word_path(word, letter_paths, spacing=0.2):
    """
    Generate a path for the entire word by combining letter paths.
    
    Args:
        word: String to generate path for
        letter_paths: Dictionary of letter paths
        spacing: Horizontal spacing between letters
    
    Returns:
        List of (x, y, z) points, with z indicating pen in/out
    """
    word_path = []
    current_x_offset = 0
    
    for i, letter in enumerate(word.lower()):
        if letter in letter_paths:
            letter_path = letter_paths[letter].copy()
            
            letter_path[:, 0] += current_x_offset
            
            first_point = letter_path[0].copy()
            word_path.append((first_point[0], first_point[1], pen_out_z))
            
            for point in letter_path:
                word_path.append((point[0], point[1], drawing_z))
            
            current_x_offset += spacing
            
            if i < len(word) - 1:
                last_point = letter_path[-1].copy()
                word_path.append((last_point[0], last_point[1], pen_out_z))
        else:
            # For unsupported characters (spaces, etc.), just move horizontally
            current_x_offset += spacing / 2
    
    return np.array(word_path)


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    model = mujoco.MjModel.from_xml_path("../universal_robots_ur5e/wooden_table_ur5e.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    site_id = model.site("attachment_site").id

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

    key_id = model.key("home").id

    mocap_id = model.body("target").mocapid[0]

    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    word_to_draw = "hello"  # Can be changed to any word with the supported letters
    letter_paths = generate_letter_paths()
    word_path = generate_word_path(word_to_draw, letter_paths)
    
    # Scale and center the path
    word_path[:, :2] = word_path[:, :2] * word_scale
    
    # Center the word on the table
    word_path[:, 0] += 0.5 - np.mean(word_path[:, 0])  # Center X
    word_path[:, 1] += 0.0 - np.mean(word_path[:, 1])  # Center Y
    
    # Define speeds for different movements
    drawing_speed = 0.1   # m/s
    pen_out_speed = 0.3   # m/s

    path_length = len(word_path)
    time_per_point = []
    
    # Calculate time for each segment based on distance and speed
    for i in range(1, path_length):
        start_point = word_path[i-1]
        end_point = word_path[i]
        
        distance = np.linalg.norm(end_point - start_point)
        
        if end_point[2] == drawing_z:
            speed = drawing_speed
        else:
            speed = pen_out_speed
            
        if distance > 1e-6:
            segment_time = distance / speed
        else:
            # THIS IS THE FIX: Assign a tiny non-zero time to zero-distance segments
            # This prevents duplicate values in cum_time.
            segment_time = 1e-5
            
        time_per_point.append(segment_time)

    cum_time = np.cumsum([0] + time_per_point)
    total_time = cum_time[-1]

    # This should now work without error
    interp_x = interp1d(cum_time, word_path[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(cum_time, word_path[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_z = interp1d(cum_time, word_path[:, 2], kind='cubic', bounds_error=False, fill_value="extrapolate")
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        downward_quat = np.array([0, 1, 0, 0])  # Pointing downward (negative z-axis)
        data.mocap_quat[mocap_id] = downward_quat
        trail_viz = RealTimeTrailVisualizer()
        while viewer.is_running():
            step_start = time.time()
            
            current_pos = data.site(site_id).xpos.copy()
            is_drawing = abs(current_pos[2] - drawing_z) < 0.01
            print(current_pos[2], drawing_z, is_drawing)

            elapsed = (data.time % (total_time * 1.5))  # Add pause between cycles
            drawing_active = elapsed < total_time
            trail_viz.add_point(current_pos, is_drawing)
            
            if drawing_active:
                target_x = float(interp_x(elapsed))
                target_y = float(interp_y(elapsed))
                target_z = float(interp_z(elapsed))
                
                data.mocap_pos[mocap_id] = np.array([target_x, target_y, target_z])
            else:
                start_x = float(interp_x(0))
                start_y = float(interp_y(0))
                start_z = pen_out_z
                data.mocap_pos[mocap_id] = np.array([start_x, start_y, start_z])

            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            mujoco.mj_step(model, data)

            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



if __name__ == "__main__":
    main()