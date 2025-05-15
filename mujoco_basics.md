# Random facts about Mujoco

### Difference between mj_step() and mj_forward()

- `mj_step()` is used to advance the simulation by one time step. It updates the state of the simulation, including the positions and velocities of the bodies, based on the applied forces and torques.
- `mj_forward()` is the same as mj_step() but does not integrate in time. It is used to update the simulation state without advancing the simulation time. 
Both functions accept the same arguments:
```python
mujoco.mj_step(model, data)
mujoco.mj_forward(model, data)
```
### 
