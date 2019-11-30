import numpy as np
import torch
import torch.nn as nn
import abc

from . import torch_utils

def gmm_loss(particle_states, log_weights, true_states):

    true_states = batch_states[:,t,:]
    assert true_states.shape == (N, state_dim)

    # Gaussian mixture model loss
    # There's probably a better way to do this with torch.distributions?
    particle_variances = torch.ones((N, state_dim), device=device)
    particle_variances[:,0] *= 0.2
    particle_variances[:,1] *= 0.2
    particle_variances[:,2] *= 0.1

    particle_squared_errors = (particles_states - true_states[:,np.newaxis,:]) ** 2
    assert particle_squared_errors.shape == (N, M, state_dim)
    log_pdfs = -0.5 * (
        torch.log(particle_variances[:,np.newaxis,:]) + \
        particle_squared_errors / particle_variances[:,np.newaxis,:]
    ).sum(axis=2)
    assert log_pdfs.shape == (N, M)
    log_pdfs = -0.5 * np.log(2 * np.pi) + log_pdfs

    # `log_pdf` should now be the probability of the true state
    # given a Gaussian centered at each particle

    # next, let's use the particle weight as our GMM priors
    log_pdfs = log_weights + log_pdfs

    # I think that's it?
    # GMM density function: p(x) = \sum_k p(x|z=k)p(z=k)
    log_beliefs = torch.logsumexp(log_pdfs, axis=1)
    assert log_beliefs.shape == (N,)

    loss = -torch.mean(log_beliefs)

class ParticleFilterDataset(torch.utils.data.Dataset):
    """
    A data preprocessor for producing particle sets from trajectories
    """
    def __init__(self, trajectories, subsequence_length=20, particle_count=200, particle_variances=None):
        """ Initialize the dataset. We chop our list of trajectories into a set of subsequences.
        Input:
          trajectories: list of trajectories, where each is a tuple of (states, observations, controls)
        """

        assert subsequence_length >= 2
        assert particle_count > 0

        subsequences = []

        # Chop up each trajectory into overlapping subsequences
        for trajectory in trajectories:
            assert len(trajectory) == 3
            states, observation, controls = trajectory

            assert len(states) == len(observation) and len(observation) == len(controls)
            trajectory_length = len(states)

            sections = trajectory_length // subsequence_length
            def split(x):
                new_length = (len(x) // subsequence_length) * subsequence_length
                x = x[:new_length]
                return np.split(x[:new_length], sections)

            for s, o, c in zip(split(states), split(observation), split(controls)):
                # Numpy => Torch
                s = torch_utils.to_torch(s)
                o = torch_utils.to_torch(o)
                c = torch_utils.to_torch(c)

                # Add to subsequences
                subsequences.append((s, o, c))

        # Create unit particle variances if no values passed in
        state_dim = subsequences[0][0].shape[1]
        if particle_variances == None:
            particle_variances = [1.] * state_dim
        assert len(particle_variances) == state_dim

        # Set properties
        self.particle_variances = particle_variances
        self.particle_count = particle_count
        self.subsequences = subsequences
        self.len = len(self.subsequences)

    def __getitem__(self, index):
        """ Get a subsequence from our dataset
        """

        states, observation, controls = self.subsequences[index]

        trajectory_length, state_dim = states.shape
        initial_state = states[0]

        # Generate noisy states as initial particles
        n = torch.distributions.Normal(torch.tensor([0.]), torch.tensor(self.particle_variances))
        initial_particles = n.sample((self.particle_count, ))
        assert initial_particles.shape == (self.particle_count, state_dim)
        initial_particles = initial_particles + initial_state

        # return image and label
        return initial_particles, states, observation, controls

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

class MeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes a likelihood given the observation.
        """
        pass


class DynamicsModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, states_prev, controls, noisy=False):
        """
        Predict the current state from the previous one + our control input.

        Parameters:
            states_prev (torch.Tensor): (N, M state_dim) states at time `t - 1`
            controls (torch.Tensor): (N, control_dim) control inputs at time `t`
            noisy (bool): whether or not we should inject noise. Typically True for particle updates.
        Returns:
            states (torch.Tensor): (N, M, state_dim) states at time `t`
        """
        pass


class ParticleFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model, soft_resample_alpha=1.0):
        super(ParticleFilterNetwork, self).__init__()
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        assert(soft_resample_alpha >= 0. and soft_resample_alpha <= 1.)
        self.soft_resample_alpha = soft_resample_alpha

    def forward(self, states_prev, log_weights_prev, observations, controls, resample=True):
        # states_prev: (N, M, *)
        # log_weights_prev: (N, M)
        # observations: (N, *)
        # controls: (N, *)
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
            log_weights = log_weights_pred - torch.logsumexp(log_weights_pred, dim=1)[:,np.newaxis]

        return best_states, states, log_weights
