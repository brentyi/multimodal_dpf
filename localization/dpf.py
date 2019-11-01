import numpy as np
import torch
import torch.nn as nn
import abc


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

            # Uniform weights
            log_weights = torch.FloatTensor(
                np.zeros_like(log_weights_pred.detach()) -
                np.log(num_particles)
            )

        return best_state, states, log_weights
