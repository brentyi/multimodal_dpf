"""
Particle Filter localization sample
author: Atsushi Sakai (@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import abc
import scipy.stats


class MeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observation, states):
        """
        For each state, computes a likelihood given the observation.
        """
        pass


class DynamicsModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, states_prev, control):
        """
        Predict the current state from the previous one + our control input.

        Parameters:
            states_prev (torch.Tensor): (N, state_dim) states at time `t - 1`
            control (torch.Tensor): (control_dim, ) control inputs at time `t`
        Returns:
            states (torch.Tensor): (N, state_dim) states at time `t`
        """
        pass


class BeaconObserver:

    def __init__(self):
        self.locations = []
        self.variances = []

    def add_beacon(self, location, variance):
        location = np.array(location)
        variance = np.array(variance)
        assert(location.shape == (2,))
        assert(variance.shape == ())

        self.locations.append(location)
        self.variances.append(variance)

    def forward(self, states, noisy=False):
        N = states.shape[0]
        num_beacons = len(self.locations)

        output = np.zeros((N, num_beacons))
        beacons = zip(self.locations, self.variances)
        for i, (location, variance) in enumerate(beacons):
            deltas = states[:, :2] - location[np.newaxis, :]
            distances = np.linalg.norm(deltas, axis=1)
            assert(distances.shape == (N,))
            output[:, i] = distances

            if noisy:
                stddev = np.sqrt(variance)
                output[:, i] += np.random.normal(scale=stddev, size=N)

        return output


class BeaconMeasurementModel(MeasurementModel):

    def __init__(self, beacon_observer):
        super(BeaconMeasurementModel, self).__init__()
        self.beacon_observer = beacon_observer

    def forward(self, observation, states):
        N = states.shape[0]
        log_likelihoods = torch.zeros(N)
        num_beacons = len(observation)
        assert(num_beacons == len(self.beacon_observer.locations))

        observations_pred = torch.from_numpy(
            self.beacon_observer.forward(states.numpy()).astype(np.float32))
        observations_error = observations_pred - observation[np.newaxis, :]

        for i in range(num_beacons):
            variance = self.beacon_observer.variances[i]
            pdf = scipy.stats.multivariate_normal.pdf(
                observations_error[:, i], cov=variance)
            assert(pdf.shape == (N,))
            log_likelihoods += torch.from_numpy(pdf.astype(np.float32))

        return log_likelihoods


class RobotDynamicsModel(DynamicsModel):

    def forward(self, states_prev, control, noisy=False):
        # Control: (v, omega)
        N = len(states_prev)
        assert(control.shape == (2,))

        v, omega = control
        if noisy:
            v_noise = np.random.normal(0, 0.1, size=N).astype(np.float32)
            v = v + torch.from_numpy(v_noise)
            omega_noise = np.random.normal(0, 0.05, size=N).astype(np.float32)
            omega = omega + torch.from_numpy(omega_noise)

        states = torch.zeros_like(states_prev)
        states[:, 0] = states_prev[:, 0] + v * np.cos(states_prev[:, 2])
        states[:, 1] = states_prev[:, 1] + v * np.sin(states_prev[:, 2])
        states[:, 2] = states_prev[:, 2] = (
            states_prev[:, 2] + omega) % (2 * np.pi)
        return states


class ParticleFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model, soft_resample_alpha):
        super(ParticleFilterNetwork, self).__init__()
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        assert(soft_resample_alpha >= 0. and soft_resample_alpha <= 1.)
        self.soft_resample_alpha = soft_resample_alpha

    def forward(self, states_prev, log_weights_prev, observation, control):
        num_particles = states_prev.shape[0]
        assert(log_weights_prev.shape == (num_particles,))

        # Particle update
        states_pred = self.dynamics_model(states_prev, control, noisy=True)

        # Particle re-weighting
        log_weights_pred = log_weights_prev + \
            self.measurement_model(observation, states_pred)

        # Find best particle
        best_index = torch.argmax(log_weights_pred)
        best_state = states_pred[best_index]

        # Re-sampling
        uniform_log_weights = torch.FloatTensor(
            np.zeros_like(log_weights_pred) - np.log(num_particles)
        )
        if self.soft_resample_alpha < 1.0:
            # Soft re-sampling
            interpolated_weights = \
                (self.soft_resample_alpha * torch.exp(log_weights_pred)) \
                + ((1. - self.soft_resample_alpha) * 1. / num_particles)

            indices = torch.multinomial(
                interpolated_weights,
                num_samples=num_particles,
                replacement=True)
            states = states_pred[indices]

            # Importance sampling & normalization
            log_weights = log_weights_pred - torch.log(interpolated_weights)

            # Normalize weights
            log_weights -= torch.logsumexp(log_weights, dim=0)
        else:
            # Standard particle filter re-sampling -- this kills gradients :(
            indices = torch.multinomial(
                torch.exp(log_weights_pred),
                num_samples=num_particles,
                replacement=True)
            states = states_pred[indices]

            # Equalize weights
            log_weights = uniform_log_weights

        return best_state, states, log_weights


iterations = 100

# (x, y, theta)
states = [np.array([0.0, 0.0, 0.0])]
controls = []
observations = []

beacon_observer = BeaconObserver()
beacon_observer.add_beacon((5, 3), 0.2)
beacon_observer.add_beacon((22, 8), 0.5)

dynamics = RobotDynamicsModel()
measurements = BeaconMeasurementModel(beacon_observer)

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

pfnet = ParticleFilterNetwork(dynamics, measurements, 1.0)
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
