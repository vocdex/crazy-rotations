"""Basic examples of forward and inverse kinematics using Mujoco"""
import numpy as np
import mujoco
import mujoco.viewer as viewer
from robot_descriptions import ur5e_mj_description

# Load the model
model = mujoco.MjModel.from_xml_path(ur5e_mj_description.MJCF_PATH)
data = mujoco.MjData(model)

# Set up initial joint positions
pi = np.pi
data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]  # Initial joint position
qpos0 = data.qpos.copy()

# Forward kinematics - updates the positions
mujoco.mj_forward(model, data)

# Get end effector position
target = data.body('wrist_3_link').xpos
print("Target =>", target)

# For visualization
print("Results")
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
init_point = data.body('wrist_3_link').xpos.copy()

# To actually visualize the simulation interactively:
with viewer.launch_passive(model, data) as viewer:
    # Set camera
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30
    
    # Run simulation loop
    for i in range(10000):  # Run for a set number of steps
        # Update visualization
        viewer.sync()
        
        # Here you could add code to update joint positions
        # data.qpos = [...] # new joint positions
        
        # Step the physics
        mujoco.mj_step(model, data)
        
        # Control the simulation speed
        viewer.wait()