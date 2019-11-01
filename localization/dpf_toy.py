import dpf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats


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


class BeaconMeasurementModel(dpf.MeasurementModel):

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


class DeepBeaconMeasurementModel(dpf.MeasurementModel):

    def __init__(self):
        super(DeepBeaconMeasurementModel, self).__init__()

        obs_dim = 2
        state_dim = 3

        Activation = nn.LeakyReLU
        hidden_units = 16
        self.observation_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_units),
            Activation(),
            # nn.Linear(hidden_units, hidden_units)
            # Activation(),
        )
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_units),
            Activation(),
            # nn.Linear(hidden_units, hidden_units)
            # Activation(),
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            Activation(),
            nn.Linear(hidden_units, hidden_units),
            Activation(),
            nn.Linear(hidden_units, hidden_units),
            Activation(),
            nn.Linear(hidden_units, 1),
            nn.LogSigmoid()
        )

    def forward(self, observation, states):
        assert(len(observation.shape) == 1)  # (obs_dim,)
        assert(len(states.shape) == 2)  # (N, state_dim)

        obs_dim = observation.shape[0]
        batch_size = states.shape[0]
        state_dim = states.shape[1]

        observation_features = self.observation_layers(
            observation[np.newaxis, :])
        state_features = self.state_layers(states)

        merged_features = torch.cat(
            (observation_features.expand(batch_size, -1), state_features),
            dim=1)

        log_likelihoods = self.shared_layers(merged_features)

        return torch.squeeze(log_likelihoods, dim=1)


class RobotDynamicsModel(dpf.DynamicsModel):

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
