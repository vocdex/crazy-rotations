import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from abc import ABC, abstractmethod
import time

# Global parameters
integration_dt = 2.0
damping = 1e-4  
dt = 0.002

class BaseIK(ABC):
    def __init__(self, model, data, site_id, tolerance=1e-4):
        self.model = model
        self.data = data
        self.site_id = site_id
        self.tolerance = tolerance
        
        # Preallocate memory
        self.jac = np.zeros((6, model.nv))
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        
        # For tracking convergence metrics
        self.error_history = []
        self.lambda_history = []  # For LM algorithm
    
    @abstractmethod
    def step(self, current_q, target_pos, target_quat):
        pass
    
    def solve(self, init_q, target_pos, target_quat, max_iterations=100, track_error=True):
        self.error_history = []
        if hasattr(self, 'lambda_history'):
            self.lambda_history = []
        
        q = init_q.copy()
        original_qpos = self.data.qpos.copy()
        
        self.data.qpos[:len(q)] = q
        mujoco.mj_forward(self.model, self.data)
        self._compute_error(target_pos, target_quat)
        error_norm = np.linalg.norm(self.error)
        
        if track_error:
            self.error_history.append(error_norm)
        
        iteration = 0
        while error_norm > self.tolerance and iteration < max_iterations:
            q = self.step(q, target_pos, target_quat)
            self._check_joint_limits(q)
            
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)
            self._compute_error(target_pos, target_quat)
            error_norm = np.linalg.norm(self.error)
            
            if track_error:
                self.error_history.append(error_norm)
            
            if error_norm < self.tolerance:
                break
            
            iteration += 1
        
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
            
        return q, iteration, error_norm
    
    def _compute_error(self, target_pos, target_quat):
        self.error_pos[:] = target_pos - self.data.site_xpos[self.site_id]
        
        site_xmat_flat = self.data.site_xmat[self.site_id].flatten()
        mujoco.mju_mat2Quat(self.site_quat, site_xmat_flat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)
    
    def _compute_jacobian(self):
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
    
    def _check_joint_limits(self, q):
        for i in range(min(len(q), self.model.nq)):
            jnt_id = i
            if jnt_id < self.model.njnt:
                lower = self.model.jnt_range[jnt_id, 0]
                upper = self.model.jnt_range[jnt_id, 1]
                if not np.isnan(lower) and not np.isnan(upper):
                    q[i] = np.clip(q[i], lower, upper)


class GradientDescentIK(BaseIK):
    def __init__(self, model, data, site_id, tolerance=1e-4, step_size=0.01):
        super().__init__(model, data, site_id, tolerance)
        self.step_size = step_size
        self.name = "Gradient Descent"
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        # Standard Jacobian transpose method
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
        self.name = "Pseudo-Inverse"
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        # Using pseudoinverse for better convergence
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
        self.name = "Gauss-Newton"
    
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
        self.name = "Levenberg-Marquardt"
        self.lambda_history = []
    
    def step(self, current_q, target_pos, target_quat):
        self.data.qpos[:len(current_q)] = current_q
        mujoco.mj_forward(self.model, self.data)
        
        self._compute_error(target_pos, target_quat)
        self._compute_jacobian()
        
        # Track lambda value
        self.lambda_history.append(self.lambda_val)
        
        current_error_norm = np.linalg.norm(self.error)
        lambda_adjusted = False
        max_attempts = 5
        
        for _ in range(max_attempts):
            JJ_T = self.jac @ self.jac.T + self.lambda_val * self.identity
            dq = self.jac.T @ np.linalg.solve(JJ_T, self.error)
            
            # Ensure dq has right dimensions
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


def analyze_single_case(model_path, seed=42):
    """
    Analyze convergence behavior for a single test case with detailed tracking.
    """
    np.random.seed(seed)  # For reproducibility
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    
    site_id = model.site("attachment_site").id
    
    solvers = [
        GradientDescentIK(model, data, site_id, tolerance=1e-4, step_size=0.5),
        PseudoInverseIK(model, data, site_id, tolerance=1e-4, step_size=0.5),
        GaussNewtonIK(model, data, site_id, tolerance=1e-4),
        LevenbergMarquardtIK(model, data, site_id, tolerance=1e-4, lambda_init=0.1)
    ]
    
    # Generate a random pose
    radius = 0.7
    target_pos = np.array([radius * 0.8, 0, radius * 0.5 + 0.2])
    
    q = Quaternion.random()
    target_quat = np.array([q.elements[3], q.elements[0], q.elements[1], q.elements[2]])
    
    init_q = np.zeros(model.nq)
    results = {}
    
    for solver in solvers:
        start_time = time.time()
        _, iterations, final_error = solver.solve(init_q, target_pos, target_quat, max_iterations=200)
        solve_time = time.time() - start_time
        
        results[solver.name] = {
            'iterations': iterations,
            'final_error': final_error,
            'time': solve_time,
            'error_history': solver.error_history,
            'lambda_history': solver.lambda_history if hasattr(solver, 'lambda_history') and len(solver.lambda_history) > 0 else None
        }
        
        print(f"{solver.name}: {iterations} iterations, final error: {final_error:.8f}, time: {solve_time:.4f}s")
    
    # Plot convergence curves
    plt.figure(figsize=(12, 8))
    
    for name, data in results.items():
        plt.semilogy(range(len(data['error_history'])), data['error_history'], label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence Comparison for Single Test Case')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot lambda adaptation for LM
    if results["Levenberg-Marquardt"]["lambda_history"] is not None:
        plt.figure(figsize=(10, 6))
        plt.semilogy(results["Levenberg-Marquardt"]["lambda_history"], label='Lambda Value')
        plt.xlabel('Iteration')
        plt.ylabel('Lambda (damping parameter)')
        plt.title('Levenberg-Marquardt Lambda Adaptation')
        plt.grid(True, which="both", ls="--")
        plt.legend()
    
    return results


def run_convergence_analysis(model_path, num_samples=20, max_iterations=200, random_seed=42):
    """
    Run a comparative analysis of different IK algorithms.
    
    Args:
        model_path: Path to the MuJoCo XML model
        num_samples: Number of random poses to test
        max_iterations: Maximum iterations per solve
        random_seed: Seed for random generator
    """
    np.random.seed(random_seed)
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    
    site_id = model.site("attachment_site").id
    
    gd_solver = GradientDescentIK(model, data, site_id, tolerance=1e-4, step_size=0.5)
    pi_solver = PseudoInverseIK(model, data, site_id, tolerance=1e-4, step_size=0.5)
    gn_solver = GaussNewtonIK(model, data, site_id, tolerance=1e-4)
    lm_solver = LevenbergMarquardtIK(model, data, site_id, tolerance=1e-4, lambda_init=0.1)
    
    solvers = [gd_solver, pi_solver, gn_solver, lm_solver]
    
    all_iterations = {solver.name: [] for solver in solvers}
    all_final_errors = {solver.name: [] for solver in solvers}
    all_error_histories = {solver.name: [] for solver in solvers}
    all_solve_times = {solver.name: [] for solver in solvers}
    success_rates = {solver.name: 0 for solver in solvers}
    
    # Generate random test cases
    for i in range(num_samples):
        print(f"Running sample {i+1}/{num_samples}")
        
        radius = 0.7
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi) + 0.5  # Offset to avoid ground
        
        target_pos = np.array([x, y, z])
        
        q = Quaternion.random()
        target_quat = np.array([q.elements[3], q.elements[0], q.elements[1], q.elements[2]])  # w,x,y,z to MuJoCo format
        
        init_q = np.random.uniform(-0.1, 0.1, model.nq)
        
        for solver in solvers:
            start_time = time.time()
            _, iterations, final_error = solver.solve(init_q, target_pos, target_quat, max_iterations)
            solve_time = time.time() - start_time
            
            all_iterations[solver.name].append(iterations)
            all_final_errors[solver.name].append(final_error)
            all_error_histories[solver.name].append(solver.error_history[:])
            all_solve_times[solver.name].append(solve_time)
            
            if final_error < solver.tolerance:
                success_rates[solver.name] += 1
    
    # Calculate success rates as percentages
    for name in success_rates:
        success_rates[name] = (success_rates[name] / num_samples) * 100
    
    # Compute average statistics
    avg_iterations = {name: np.mean(iterations) for name, iterations in all_iterations.items()}
    avg_final_errors = {name: np.mean(errors) for name, errors in all_final_errors.items()}
    avg_solve_times = {name: np.mean(times) for name, times in all_solve_times.items()}
    
    # Print statistics
    print("\nAlgorithm Performance Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Avg Iterations':<15} {'Avg Final Error':<20} {'Avg Time (s)':<15} {'Success Rate (%)':<15}")
    print("-" * 80)
    for name in avg_iterations.keys():
        print(f"{name:<20} {avg_iterations[name]:<15.2f} {avg_final_errors[name]:<20.8f} {avg_solve_times[name]:<15.4f} {success_rates[name]:<15.1f}")
    
    plt.figure(figsize=(12, 8))
    
    for solver in solvers:
        name = solver.name
        max_len = max(len(hist) for hist in all_error_histories[name])
        
        padded_histories = []
        for hist in all_error_histories[name]:
            if len(hist) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(hist)] = hist
                padded_histories.append(padded)
            else:
                padded_histories.append(hist)
        
        histories_array = np.array(padded_histories)
        mean_history = np.nanmean(histories_array, axis=0)
        plt.semilogy(mean_history, label=name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title(f'Average Convergence Behavior (n={num_samples} random poses)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    
    # Return the collected statistics
    return {
        'iterations': all_iterations,
        'errors': all_final_errors,
        'times': all_solve_times,
        'histories': all_error_histories,
        'success_rates': success_rates
    }


if __name__ == "__main__":
    model_path = "../mujoco_menagerie/universal_robots_ur5e/scene.xml"
    
    run_type = "benchmark"  # Options: "single_case" or "benchmark"
    
    if run_type == "single_case":
        results = analyze_single_case(model_path, seed=42)
    else:
        results = run_convergence_analysis(
            model_path, 
            num_samples=50,  # Number of random poses to test
            max_iterations=200,  # Maximum iterations per solve
            random_seed=42  # For reproducibility
        )
    
    plt.show()