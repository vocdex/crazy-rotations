import mujoco
import mujoco.viewer 
import numpy as np
import time
from pyquaternion import Quaternion
from abc import ABC, abstractmethod

# Global parameters
integration_dt = 2.0
damping = 1e-4  
gravity_compensation = True  # Enable gravity compensation for better control
dt = 0.002


class BaseIK(ABC):
    def __init__(self, model, data, site_id, tolerance=1e-4):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.tolerance = tolerance
        self.viewer = None
        
        # Preallocate memory
        self.jac = np.zeros((6, model.nv))
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
    
    @abstractmethod
    def step(self, current_q, target_pos, target_quat):
        pass
    
    def solve(self, init_q, target_pos, target_quat, max_iterations=100):
        q = init_q.copy()
        original_qpos = self.data.qpos.copy()
        
        self.data.qpos[:len(q)] = q
        mujoco.mj_forward(self.model, self.data)
        self._compute_error(target_pos, target_quat)
        error_norm = np.linalg.norm(self.error)
        
        iteration = 0
        while error_norm > self.tolerance and iteration < max_iterations:
            q = self.step(q, target_pos, target_quat)
            self._check_joint_limits(q)
            
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)
            self._compute_error(target_pos, target_quat)
            error_norm = np.linalg.norm(self.error)
            
            if error_norm < self.tolerance:
                print("Tolerance reached")
                break
            iteration += 1
        
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
            
        return q
    
    def _compute_error(self, target_pos, target_quat):
        self.error_pos[:] = target_pos - self.data.site_xpos[self.site_id]
        
        site_xmat_flat = self.data.site_xmat[self.site_id].flatten()
        mujoco.mju_mat2Quat(self.site_quat, site_xmat_flat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)
    
    def _compute_jacobian(self):
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
    
    def _set_control(self, target_q):
        for i in range(min(len(target_q), self.model.nu)):
            self.data.ctrl[i] = target_q[i]
    
    def _check_joint_limits(self, q):
        for i in range(min(len(q), self.model.nq)):
            jnt_id = i
            if jnt_id < self.model.njnt:
                lower = self.model.jnt_range[jnt_id, 0]
                upper = self.model.jnt_range[jnt_id, 1]
                if not np.isnan(lower) and not np.isnan(upper):
                    q[i] = np.clip(q[i], lower, upper)
    
    def _apply_gravity_compensation(self):
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = 0
        self.data.qfrc_applied[:self.model.nv] = -self.data.qfrc_bias
    
    def _add_target_visualization(self, target_pos, target_quat):
        if self.model.nmocap > 0:
            mocap_id = 0
            self.data.mocap_pos[mocap_id] = target_pos
            self.data.mocap_quat[mocap_id] = target_quat
        elif self.viewer:
            self.viewer.add_marker(
                pos=target_pos,
                size=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 0.5],
                type=mujoco.mjtGeom.mjGEOM_SPHERE
            )
            
            axis_length = 0.05
            rot_mat = np.zeros((3, 3))
            mujoco.mju_quat2Mat(rot_mat.flatten(), target_quat)
            
            for i, color in enumerate([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]):
                axis = rot_mat[:, i] * axis_length
                self.viewer.add_marker(
                    pos=target_pos,
                    end=target_pos + axis,
                    size=[0.001],
                    rgba=color,
                    type=mujoco.mjtGeom.mjGEOM_LINE
                )

    def run_ik_with_visualization(self, init_q, target_pos, target_quat,
                            mode="simulation", max_iterations=100,
                            duration=1.0, dt=0.01, debug=False):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            self.viewer = viewer
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            self._add_target_visualization(target_pos, target_quat)
            
            if mode == "simulation":
                self.data.qpos[:len(init_q)] = init_q
                mujoco.mj_forward(self.model, self.data)
                n_steps = int(duration / dt)
                
                for i in range(n_steps):
                    if not self.viewer.is_running():
                        break
                    
                    step_start = time.time()
                    current_q = self.data.qpos[:len(init_q)].copy()
                    
                    self._compute_error(target_pos, target_quat)
                    self._compute_jacobian()
                    
                    q_target = self.step(current_q, target_pos, target_quat)
                    self._check_joint_limits(q_target)
                    self._set_control(q_target)
                    
                    if gravity_compensation:
                        self._apply_gravity_compensation()
                    
                    mujoco.mj_step(self.model, self.data)
                    self.viewer.sync()
                    
                    time_until_next_step = dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                
                return self.data.qpos[:len(init_q)].copy()
                
            elif mode == "convergence":
                q = init_q.copy()
                self.data.qpos[:len(q)] = q
                mujoco.mj_forward(self.model, self.data)
                self._compute_error(target_pos, target_quat)
                error_norm = np.linalg.norm(self.error)
                
                if debug:
                    print(f"Initial error: {error_norm:.6f}")
                
                self.viewer.sync()
                time.sleep(0.5)
                
                iteration = 0
                while error_norm > self.tolerance and iteration < max_iterations:
                    q = self.step(q, target_pos, target_quat)
                    self._check_joint_limits(q)
                    
                    self.data.qpos[:len(q)] = q
                    mujoco.mj_forward(self.model, self.data)
                    self._compute_error(target_pos, target_quat)
                    error_norm = np.linalg.norm(self.error)
                    
                    if debug:
                        print(f"Iteration {iteration}, error: {error_norm:.6f}")
                    
                    self.viewer.sync()
                    time.sleep(0.05)
                    
                    iteration += 1
                    
                    if not self.viewer.is_running():
                        break
                
                if debug:
                    print(f"Final error after {iteration} iterations: {error_norm:.10f}")
                
                time.sleep(0.5)
                return q
            else:
                raise ValueError(f"Unknown visualization mode: {mode}")


class GradientDescentIK(BaseIK):
    def __init__(self, model, data, site_id, tolerance=1e-4, step_size=0.01):
        super().__init__(model, data, site_id, tolerance)
        self.step_size = step_size
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        gradient = self.jac.T @ self.error
        
        if len(gradient) > len(current_q):
            gradient = gradient[:len(current_q)]
        elif len(gradient) < len(current_q):
            padded_gradient = np.zeros_like(current_q)
            padded_gradient[:len(gradient)] = gradient
            gradient = padded_gradient
        
        return current_q + self.step_size * gradient


class PseudoInverseIK(BaseIK):
    def __init__(self, model, data, site_id, tolerance=1e-4, step_size=0.1):
        super().__init__(model, data, site_id, tolerance)
        self.step_size = step_size
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        dq = np.linalg.pinv(self.jac) @ self.error
        
        if len(dq) > len(current_q):
            dq = dq[:len(current_q)]
        elif len(dq) < len(current_q):
            padded_dq = np.zeros_like(current_q)
            padded_dq[:len(dq)] = dq
            dq = padded_dq
        
        return current_q + self.step_size * dq


class GaussNewtonIK(BaseIK):
    def __init__(self, model, data, site_id, tolerance=1e-4):
        super().__init__(model, data, site_id, tolerance)
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        JJ_T = self.jac @ self.jac.T + damping * np.eye(self.jac.shape[0])
        dq = self.jac.T @ np.linalg.solve(JJ_T, self.error)
        
        if len(dq) > len(current_q):
            dq = dq[:len(current_q)]
        elif len(dq) < len(current_q):
            padded_dq = np.zeros_like(current_q)
            padded_dq[:len(dq)] = dq
            dq = padded_dq
        
        return current_q + dq


class LevenbergMarquardtIK(BaseIK):
    def __init__(self, model, data, site_id, tolerance=1e-4, lambda_init=0.01,
                lambda_factor=10.0, min_lambda=1e-6, max_lambda=1e3):
        super().__init__(model, data, site_id, tolerance)
        self.lambda_val = lambda_init
        self.lambda_factor = lambda_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.identity = np.eye(6)
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        current_error_norm = np.linalg.norm(self.error)
        lambda_adjusted = False
        max_attempts = 5
        
        for _ in range(max_attempts):
            JJ_T = self.jac @ self.jac.T + self.lambda_val * self.identity
            dq = self.jac.T @ np.linalg.solve(JJ_T, self.error)
            
            if len(dq) > len(current_q):
                dq = dq[:len(current_q)]
            elif len(dq) < len(current_q):
                padded_dq = np.zeros_like(current_q)
                padded_dq[:len(dq)] = dq
                dq = padded_dq
            
            q_new = current_q + dq
            self._check_joint_limits(q_new)
            
            self.data.qpos[:len(q_new)] = q_new
            mujoco.mj_forward(self.model, self.data)
            self._compute_error(target_pos, target_quat)
            new_error_norm = np.linalg.norm(self.error)
            
            if new_error_norm < current_error_norm:
                self.lambda_val = max(self.min_lambda, self.lambda_val / self.lambda_factor)
                lambda_adjusted = True
                break
            else:
                self.lambda_val = min(self.max_lambda, self.lambda_val * self.lambda_factor)
        
        if not lambda_adjusted:
            q_new = current_q + dq
            self._check_joint_limits(q_new)
        
        return q_new


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inverse Kinematics Solver for UR5e")
    parser.add_argument("--solver", type=str, default="lm", choices=["gd", "pi", "gn", "lm"],
                        help="IK solver type: 'gd' for Gradient Descent, 'pi' for Pseudo-Inverse, 'gn' for Gauss-Newton, 'lm' for Levenberg-Marquardt")
    parser.add_argument("--mode", type=str, default="simulation", choices=["simulation", "convergence"],
                        help="Mode of operation: 'simulation' or 'convergence'")
    parser.add_argument("--random-quat", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path("../mujoco_menagerie/universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)
    
    model.opt.timestep = dt
    site_id = model.site("attachment_site").id
    target_pos = np.array([0.5, 0.5, 0.5])
    if args.random_quat:
        q = Quaternion.random() # PyQuaternion uniformly samples from the unit sphere
        target_quat = np.array([q.elements[3], q.elements[0], q.elements[1], q.elements[2]])
    else:
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    init_q = np.zeros(model.nq)
    
    if args.solver == "gd":
        ik_solver = GradientDescentIK(model, data, site_id, tolerance=1e-4, step_size=0.5)
    elif args.solver == "pi":
        ik_solver = PseudoInverseIK(model, data, site_id, tolerance=1e-4, step_size=0.5)
    elif args.solver == "gn":
        ik_solver = GaussNewtonIK(model, data, site_id, tolerance=1e-4)
    else:  # Default to Levenberg-Marquardt
        ik_solver = LevenbergMarquardtIK(model, data, site_id, tolerance=1e-4, lambda_init=0.1)
    
    # Run IK with visualization
    print(f"Running {args.solver} solver in {args.mode} mode")
    print(f"Target position: {target_pos}")
    print(f"Target orientation: {target_quat}")
    
    final_q = ik_solver.run_ik_with_visualization(
        init_q,
        target_pos,
        target_quat,
        duration=5.0,  # For simulation mode
        mode=args.mode,
        max_iterations=1000,  # For convergence mode
        debug=True
    )
    
    print("Final joint angles:", final_q)