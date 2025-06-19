"""The code is adapted from Kevin Zakka's MJCTRL codebase:
https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
from traj_visualizer import RealTimeTrailVisualizer

integration_dt: float = 1.0

damping: float = 1e-4
gravity_compensation: bool = False
dt: float = 0.002
max_angvel = 0.0

table_height = 0.30
drawing_z = table_height + 0.05  
pen_out_z = table_height + 0.10  

radius = 0.05      # Smaller radius
center_x = 0.6     
center_y = 0.0
frequency = 0.2 
def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    model = mujoco.MjModel.from_xml_path("../universal_robots_ur5e/wooden_table_ur5e.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    site_id = model.site("attachment_site").id

    # Name of bodies we wish to apply gravity compensation to.
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

    def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
        """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
        as a function of time t and frequency f."""
        x = r * np.cos(2 * np.pi * f * t) + h
        y = r * np.sin(2 * np.pi * f * t) + k
        return np.array([x, y])

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        trail_viz = RealTimeTrailVisualizer()

        # Enable contact visualization to debug
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

        while viewer.is_running():
            step_start = time.time()
            current_pos = data.site(site_id).xpos.copy()
            is_drawing = True

            trail_viz.add_point(current_pos, is_drawing)

            data.mocap_pos[mocap_id, 0:2] = circle(data.time, radius, center_x, center_y, frequency)
            data.mocap_pos[mocap_id, 2] = drawing_z
            data.mocap_quat[mocap_id] = [0,1,0,0]  # Facing downwards
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
