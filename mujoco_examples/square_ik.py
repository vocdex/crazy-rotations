"""
This script uses the MuJoCo physics engine to control a UR5e robot arm to draw a square shape on 3D horizontal plane.
The code is adapted from Kevin Zakka's MJCTRL codebase:
https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds
integration_dt: float = 1.0

# Damping term for the pseudoinverse
damping: float = 1e-4

# Whether to enable gravity compensation
gravity_compensation: bool = False

# Simulation timestep in seconds
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable
max_angvel = 0.0


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data
    model = mujoco.MjModel.from_xml_path("../mujoco_menagerie/universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # Override the simulation timestep
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

    # Define a square trajectory for the end-effector to follow
    def square_path(t: float, side_length: float, center_x: float, center_y: float, cycle_time: float) -> np.ndarray:
        """
        Return the (x, y) coordinates of a square path with side length centered at (center_x, center_y)
        as a function of time t and cycle_time (time to complete one full cycle).
        
        The square is traversed in counter-clockwise direction starting from the bottom-right corner.
        """
        # Normalize time to [0, 1] for one complete cycle
        normalized_time = (t % cycle_time) / cycle_time
        
        # Each side takes up 0.25 of the normalized time
        half_side = side_length / 2
        
        if normalized_time < 0.25:
            # Bottom side (right to left)
            progress = normalized_time * 4
            x = center_x + half_side - side_length * progress
            y = center_y - half_side
        elif normalized_time < 0.5:
            # Left side (bottom to top)
            progress = (normalized_time - 0.25) * 4
            x = center_x - half_side
            y = center_y - half_side + side_length * progress
        elif normalized_time < 0.75:
            # Top side (left to right)
            progress = (normalized_time - 0.5) * 4
            x = center_x - half_side + side_length * progress
            y = center_y + half_side
        else:
            # Right side (top to bottom)
            progress = (normalized_time - 0.75) * 4
            x = center_x + half_side
            y = center_y + half_side - side_length * progress
            
        return np.array([x, y])

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Parameters for the square path
        square_side_length = 0.2  # Side length of square in meters
        square_center_x = 0.5     # X-coordinate of square center
        square_center_y = 0.0     # Y-coordinate of square center
        cycle_time = 8.0          # Time to complete one full cycle in seconds
        
        # Fixed z-coordinate for the path
        z_height = 0.5

        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site
            xy_pos = square_path(data.time, square_side_length, square_center_x, square_center_y, cycle_time)
            data.mocap_pos[mocap_id, 0] = xy_pos[0]
            data.mocap_pos[mocap_id, 1] = xy_pos[1]
            data.mocap_pos[mocap_id, 2] = z_height

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