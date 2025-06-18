"""
This script uses the MuJoCo physics engine to control a UR5e robot arm to draw a square shape on 3D horizontal plane.
The code is adapted from Kevin Zakka's MJCTRL codebase:
https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from traj_visualizer import RealTimeTrailVisualizer

integration_dt: float = 5.0
damping: float = 1e-3
gravity_compensation: bool = True  # Enable for better tracking
dt: float = 0.002
max_angvel = 0.0  # Enable velocity limiting for smoother motion

# Table parameters
table_height = 0.30
drawing_z = table_height + 0.01  # Just above the table surface
pen_out_z = table_height + 0.1   # Lifted position above table

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

    def square_path_with_pen(t: float, side_length: float, center_x: float, center_y: float, cycle_time: float) -> tuple:
        """
        Return the (x, y, z) coordinates and pen state for drawing a square.
        Returns: (x, y, z, is_drawing)
        """
        # Normalize time to [0, 1] for one complete cycle
        normalized_time = (t % cycle_time) / cycle_time
        
        half_side = side_length / 2
        
        # Add transition phases between drawing segments
        drawing_phase_duration = 0.2  # 20% of cycle time per side
        transition_duration = 0.05    # 5% transition between sides
        
        if normalized_time < drawing_phase_duration:
            # Drawing bottom side (right to left)
            progress = normalized_time / drawing_phase_duration
            x = center_x + half_side - side_length * progress
            y = center_y - half_side
            z = drawing_z
            is_drawing = True
            
        elif normalized_time < drawing_phase_duration + transition_duration:
            # Transition to left side
            x = center_x - half_side
            y = center_y - half_side
            z = pen_out_z
            is_drawing = False
            
        elif normalized_time < 2 * drawing_phase_duration + transition_duration:
            # Drawing left side (bottom to top)
            progress = (normalized_time - drawing_phase_duration - transition_duration) / drawing_phase_duration
            x = center_x - half_side
            y = center_y - half_side + side_length * progress
            z = drawing_z
            is_drawing = True
            
        elif normalized_time < 2 * drawing_phase_duration + 2 * transition_duration:
            # Transition to top side
            x = center_x - half_side
            y = center_y + half_side
            z = pen_out_z
            is_drawing = False
            
        elif normalized_time < 3 * drawing_phase_duration + 2 * transition_duration:
            # Drawing top side (left to right)
            progress = (normalized_time - 2 * drawing_phase_duration - 2 * transition_duration) / drawing_phase_duration
            x = center_x - half_side + side_length * progress
            y = center_y + half_side
            z = drawing_z
            is_drawing = True
            
        elif normalized_time < 3 * drawing_phase_duration + 3 * transition_duration:
            # Transition to right side
            x = center_x + half_side
            y = center_y + half_side
            z = pen_out_z
            is_drawing = False
            
        elif normalized_time < 4 * drawing_phase_duration + 3 * transition_duration:
            # Drawing right side (top to bottom)
            progress = (normalized_time - 3 * drawing_phase_duration - 3 * transition_duration) / drawing_phase_duration
            x = center_x + half_side
            y = center_y + half_side - side_length * progress
            z = drawing_z
            is_drawing = True
            
        else:
            # Final transition back to start
            x = center_x + half_side
            y = center_y - half_side
            z = pen_out_z
            is_drawing = False
            
        return x, y, z, is_drawing

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        downward_quat = np.array([0, 1, 0, 0])  # Pointing downward (negative z-axis)
        data.mocap_quat[mocap_id] = downward_quat

        square_side_length = 0.15  # Smaller square for better visualization
        square_center_x = 0.5      # X-coordinate of square center
        square_center_y = 0.0      # Y-coordinate of square center
        cycle_time = 30.0          # Slower for clearer drawing
        
        trail_viz = RealTimeTrailVisualizer()

        while viewer.is_running():
            step_start = time.time()

            # Get target position and drawing state
            x, y, z, expected_drawing = square_path_with_pen(
                data.time, square_side_length, square_center_x, square_center_y, cycle_time
            )
            
            # Set mocap target
            data.mocap_pos[mocap_id] = np.array([x, y, z])

            # Get current end-effector position
            current_pos = data.site(site_id).xpos.copy()
            
            # Check if we're actually drawing (close to drawing height)
            is_drawing = abs(current_pos[2] - drawing_z) < 0.008  # Slightly larger tolerance
            
            # Update trail visualization
            trail_viz.add_point(current_pos, is_drawing)
            
            # Print status occasionally
            if int(data.time * 10) % 50 == 0:  # Every 5 seconds
                print(f"Time: {data.time:.1f}s, Expected drawing: {expected_drawing}, "
                      f"Actual drawing: {is_drawing}, Height: {current_pos[2]:.3f}")

            # Control computation (unchanged)
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