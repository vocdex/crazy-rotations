"""Given the pose of an object, move the end effector to pre-grasp pose, and then grasp the object to place it at a new location.
The inverse kinematics is already implemnted """
import mujoco
import mujoco.viewer 
import numpy as np
import time
from pyquaternion import Quaternion
from inverse_kinematics import LevenbergMarquardtIK


