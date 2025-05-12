"""
Simple Pendulum Simulation with Interactive Visualization
"""
import mujoco
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from robot_descriptions.loaders.mujoco import load_robot_description

model = load_robot_description("ur5e_mj_description")

data = mujoco.MjData(model)
duration = 5.0  # seconds
framerate = 60
height = 480
width = 640
frames = []
renderer = mujoco.Renderer(model, height, width)
mujoco.mj_resetData(model, data)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model,camera)
camera.distance = 2.0

#Put a position of the joints to get a test point
pi = np.pi
data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0]

#Initial joint position
qpos0 = data.qpos.copy()

#Step the simulation.
mujoco.mj_forward(model, data)

#Use the last piece as an "end effector" to get a test point in cartesian 
# coordinates
target = data.body('wrist_3_link').xpos
print("Target =>",target)

#Plot results
print("Results")
mujoco.mj_resetDataKeyframe(model, data, 1)
mujoco.mj_forward(model, data)
init_point = data.body('wrist_3_link').xpos.copy()
renderer.update_scene(data, camera)
target_plot = renderer.render()

data.qpos = qpos0
mujoco.mj_forward(model, data)
result_point = data.body('wrist_3_link').xpos.copy()
renderer.update_scene(data, camera)
result_plot = renderer.render()

print("initial point =>", init_point)
print("Desire point =>", result_point, "\n")

images = {
    'Initial position': target_plot,
    ' Desire end effector position': result_plot,
}
# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Plot the images
for i, (title, img) in enumerate(images.items()):
    ax[i].imshow(img)
    ax[i].set_title(title)
    ax[i].axis('off')
# Show the plot
plt.tight_layout()
plt.show()


# import time

# import mujoco
# import mujoco.viewer

# m = mujoco.MjModel.from_xml_path('../mujoco_menagerie/franka_emika_panda/scene.xml')
# d = mujoco.MjData(m)
# with mujoco.viewer.launch_passive(m, d) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   while viewer.is_running() and time.time() - start < 30:
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(m, d)

#     # Example modification of a viewer option: toggle contact points every two seconds.
#     with viewer.lock():
#       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = m.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)