"""
Particle Filter localization sample
author: Atsushi Sakai (@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import dpf
import dpf_toy


iterations = 100

# (x, y, theta)
states = [np.array([0.0, 0.0, 0.0])]
controls = []
observations = []

beacon_observer = dpf_toy.BeaconObserver()
beacon_observer.add_beacon((5, 3), 0.2)
beacon_observer.add_beacon((22, 8), 0.5)

dynamics = dpf_toy.RobotDynamicsModel()
measurements = dpf_toy.BeaconMeasurementModel(beacon_observer)

# Simulation
for _ in range(iterations):
    control = torch.from_numpy(np.random.uniform(
        low=[0, -0.1], high=[1, 0.1], size=(2,)).astype(np.float32))
    new_state = dynamics.forward(
        torch.from_numpy(states[-1][np.newaxis, :].astype(np.float32)), control, noisy=True)

    states.append(new_state[0].numpy())
    controls.append(control.numpy())
true_states = np.array(states)

# Dead-reckoning
states = [true_states[0]]
for control in controls:
    new_state = dynamics.forward(
        torch.from_numpy(states[-1][np.newaxis, :].astype(np.float32)), control, noisy=False)
    states.append(new_state[0].numpy())
dead_reckoned_states = np.array(states)

# Particle filter network
observations = beacon_observer.forward(true_states[1:])
states = [true_states[0]]
num_particles = 100
particle_states = torch.FloatTensor(
    [true_states[0] for _ in range(num_particles)])
particle_weights = torch.ones(num_particles)

pfnet = dpf.ParticleFilterNetwork(dynamics, measurements, 1.0)
for control, observation in zip(controls, observations):
    # Type conversions
    observation = torch.from_numpy(observation.astype(np.float32))
    control = torch.from_numpy(control.astype(np.float32))

    # Particle filter network: forward
    best_state, particle_states, particle_weights = pfnet.forward(
        particle_states, particle_weights, observation, control)

    states.append(best_state.numpy())
pf_states = np.array(states)

# Plot trajectories
plt.scatter(dead_reckoned_states[:, 0],
            dead_reckoned_states[:, 1], marker=".", label="Dead-reckoned")
plt.scatter(true_states[:, 0], true_states[:, 1],
            marker=".", label="Ground-truth")
plt.scatter(pf_states[:, 0], pf_states[:, 1],
            marker=".", label="Particle Filter")

pf_states = particle_states.numpy()
plt.scatter(pf_states[:, 0], pf_states[:, 1], marker=".", label="Particles")

# Plot beacons
beacon_locations = np.asarray(beacon_observer.locations)
plt.scatter(beacon_locations[:, 0], beacon_locations[:, 1], label="Beacons")

plt.legend()
plt.show()


#
# class MultiBeacons:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     def observation(self, state):
#         # State is: (x, y, theta)
#         if state.shape == (3,):
#             # Single input
#             return self.
#         elif len(state.shape) == 2 and state.shape[1] == 3:
#             # Batched input
#
#         else:
#             assert(False)
