# crazy-rotations
This is a collection of scripts to learn the basics of robotics using MuJoco physics engine.

Here's are the notes on various topics:

- [Inverse Kinematics](./inverse_kinematics.md)
- [MuJoco Basics](./mujoco_basics.md)
- [Rotation Representations](./rotation_representations.md)



## Mujoco Examples
The goal of this project is to learn basic pick-and-place tasks with robotic arms using MuJoco physics engine.

We start simple and then move to more complex tasks.
The object is a rigid box for simplicity. The robot arm is ur5e model from Mujoco-Menagerie.
- The object pose is fixed and given, and the robot arm is moved to pick it up(pre-grasp, grasp, post-grasp).
- The object pose is obtained via fiducial markers, and the robot arm is moved to pick it up(pre-grasp, grasp, post-grasp).
- The object pose is obtained via RGBD-based pose estimators, and the robot arm is moved to pick it up(pre-grasp, grasp, post-grasp).

The object has arbitrary shape and we use grasp detectors (AnyGrasp, GraspIt) to find the best grasp pose.
- The object has arbitrary shape and we use grasp detectors (AnyGrasp, GraspIt) to find the best grasp pose.
- Hook up VLMs to do common-sense visual reasoning to choose the best grasp candidates.
