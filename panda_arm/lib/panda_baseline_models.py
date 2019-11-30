import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resblocks


class PandaBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

        units = 16
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units // 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(units // 2),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units // 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(units // 2),
        )
        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=4),
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
        self.shared_layers = nn.Sequential(
            nn.Linear((units // 2) * 2 + units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, 2),  # Directly output new state
            # nn.LogSigmoid()
        )

    def forward(self, states_prev, observations, controls):
        assert type(observations) == dict  # (N, {})
        assert len(states.shape) == 2  # (N, state_dim)
        assert len(controls.shape) == 2  # (N, control_dim,)

        N, state_dim = states.shape
        N, control_dim = controls.shape

        # Construct state features
        state_features = self.state_layers(states)

        # Construct observation features
        # (N, obs_dim)
        observation_features = torch.cat((
            self.observation_image_layers(observations['image']),
            self.observation_pose_layers(observations['gripper_pose']),
            self.observation_sensors_layers(observations['gripper_sensors']),
        ), dim=1)

        # Construct control features
        control_features = self.control_layers(controls)

        # Merge features & regress next state
        merged_features = torch.cat((
            state_features,
            observation_features,
            control_features
        ), dim=1)
        assert len(merged_features.shape) == 2  # (N, feature_dim)
        assert merged_features.shape[0] == N
        new_state = self.shared_layers(merged_features)
        return new_state
