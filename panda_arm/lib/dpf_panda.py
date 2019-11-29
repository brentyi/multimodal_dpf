import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resblocks
from . import file_utils
from . import dpf


class PandaDynamicsModel(dpf.DynamicsModel):

    def __init__(self, state_noise=(0.1, 0.05)):
        super().__init__()

        state_dim = 3
        control_dim = 20

        self.state_noise = state_noise

        units = 16
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units // 2),
            resblocks.Linear(units // 2),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units // 2),
            resblocks.Linear(units // 2),
        )
        self.shared_layers = nn.Sequential(
            resblocks.Linear(units),
            resblocks.Linear(units, bottleneck_units=units // 2),
            resblocks.Linear(units),
            nn.Linear(units, state_dim),
        )

        self.units = units

    def forward(self, states_prev, controls, noisy=False):
        # states_prev:  (N, M, state_dim)
        # controls: (N, control_dim)

        assert(len(states_prev.shape) == 3)  # (N, M, state_dim)
        assert(len(controls.shape) == 2)  # (N, control_dim,)

        # N := distinct trajectory count
        # M := particle count
        N, M, state_dim = states_prev.shape

        # (N, control_dim) => (N, units // 2)
        control_features = self.control_layers(controls)
        assert control_features.shape == (N, self.units // 2)

        # (N, units // 2) => (N, M, units // 2)
        control_features = control_features[:, np.newaxis, :].expand(
            N, M, self.units // 2)
        assert control_features.shape == (N, M, self.units // 2)

        # (N, M, state_dim) => (N, M, units // 2)
        state_features = self.state_layers(states_prev)
        assert state_features.shape == (N, M, self.units // 2)

        # (N, M, units)
        merged_features = torch.cat(
            (control_features, state_features),
            dim=2)
        assert merged_features.shape == (N, M, self.units)

        # (N, M, units * 2) => (N, M, 1)
        state_update = self.shared_layers(merged_features)
        assert state_update.shape == (N, M, state_dim)

        # Compute new states
        states_new = states_prev + state_update
        assert states_new.shape == (N, M, state_dim)

        # Add noise if desired
        if noisy:
            dist = torch.distributions.Normal(
                torch.tensor([0.]), torch.tensor(self.state_noise))
            noise = dist.sample((N, M)).to(states_new.device)
            assert noise.shape == (N, M, state_dim)
            states_new = states_new + noise

        # Return (N, M, state_dim)
        return states_new


class PandaMeasurementModel(dpf.MeasurementModel):

    def __init__(self, units=16):
        super().__init__()

        obs_pose_dim = 7
        obs_sensors_dim = 7
        state_dim = 2

        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=3),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 16 * 16 = 256
            nn.Linear(256, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pose_dim, units),
            resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(units),
        )
        self.state_layers = nn.Sequential(
            nn.Identity(),
        )

        self.shared_observation_layers = nn.Sequential(
            nn.Linear(units * 3 + state_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, 1),
            # nn.LogSigmoid()
        )

        self.units = units

    def forward(self, observations, states):
        assert(type(observations.shape) == dict)
        assert(len(states.shape) == 3)  # (N, M, state_dim)

        # N := distinct trajectory count
        # M := particle count

        N, M, _ = states.shape

        # Construct observations feature vector
        # (N, obs_dim)
        observation_features = torch.cat((
            self.observation_image_layers(observations['image']),
            self.observation_pose_layers(observations['gripper_pose']),
            self.observation_sensors_layers(observations['gripper_sensors']),
        ), dim=1)

        # (N, obs_dim) => (N, M, obs_dim)
        observation_features = observation_features[:, np.newaxis, :].expand(
            N, M, self.units * 3)
        assert observation_features.shape == (N, M, self.units * 3)

        # (N, M, state_dim) => (N, M, units)
        state_features = self.state_layers(states)
        assert state_features.shape == (N, M, self.units)

        # (N, M, units)
        merged_features = torch.cat(
            (observation_features, state_features),
            dim=2)
        assert merged_features.shape == (N, M, self.units * 4)

        # (N, M, units * 4) => (N, M, 1)
        log_likelihoods = self.shared_layers(merged_features)
        assert log_likelihoods.shape == (N, M, 1)

        # Return (N, M)
        return torch.squeeze(log_likelihoods, dim=2)
