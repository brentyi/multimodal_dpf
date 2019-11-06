import numpy as np
import torch
import torch.nn as nn
import abc


class MeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes a likelihood given the observation.
        """
        pass


class DynamicsModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, states_prev, controls):
        """
        Predict the current state from the previous one + our control input.

        Parameters:
            states_prev (torch.Tensor): (N, M state_dim) states at time `t - 1`
            controls (torch.Tensor): (N, control_dim) control inputs at time `t`
        Returns:
            states (torch.Tensor): (N, M, state_dim) states at time `t`
        """
        pass


class ParticleFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model, soft_resample_alpha):
        super(ParticleFilterNetwork, self).__init__()
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        assert(soft_resample_alpha >= 0. and soft_resample_alpha <= 1.)
        self.soft_resample_alpha = soft_resample_alpha

    def forward(self, states_prev, log_weights_prev, observations, controls, resample=True):
        # states_prev: (N, M, state_dim)
        # log_weights_prev: (N, M)
        # observations: (N, obs_dim)
        # controls: (N, control_dim)
        #
        # N := distinct trajectory count
        # M := particle count

        N, M, _ = states_prev.shape
        assert log_weights_prev.shape == (N, M)

        # Particle update
        states_pred = self.dynamics_model(states_prev, controls, noisy=True)

        # Particle re-weighting
        log_weights_pred = log_weights_prev + \
            self.measurement_model(observations, states_pred)

        # Find best particle
        log_weights_pred = log_weights_pred - torch.logsumexp(log_weights_pred, dim=1)[:,np.newaxis]
        best_states = torch.sum(torch.exp(log_weights_pred[:,:,np.newaxis]) * states_pred, dim=1)

        # Re-sampling
        if resample:
            if self.soft_resample_alpha < 1.0:
                # This still needs to be re-adapted for the new minibatch shape
                assert False

                # Soft re-sampling
                interpolated_weights = \
                    (self.soft_resample_alpha * torch.exp(log_weights_pred)) \
                    + ((1. - self.soft_resample_alpha) / M)

                indices = torch.multinomial(
                    interpolated_weights,
                    num_samples=M,
                    replacement=True)
                states = states_pred[indices]

                # Importance sampling & normalization
                log_weights = log_weights_pred - torch.log(interpolated_weights)

                # Normalize weights
                log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            else:
                # Standard particle filter re-sampling -- this kills gradients :(
                states = torch.zeros_like(states_pred)
                for i in range(N):
                    indices = torch.multinomial(
                        torch.exp(log_weights_pred[i]),
                        num_samples=M,
                        replacement=True)
                    states[i] = states_pred[i][indices]

                # Uniform weights
                log_weights = torch.zeros_like(log_weights_pred) - np.log(M)
        else:
            # Just use predicted states as output
            states = states_pred

            # Normalize predicted weights
            log_weights = log_weights_pred - torch.logsumexp(log_weights_pred, dim=0)

        return best_states, states, log_weights
