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

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, states, noisy=False):
        #  Reshape arbitrary dimensions to (N, state_dim)
        states = np.asarray(states)
        orig_shape = states.shape
        state_dim = orig_shape[-1]
        states = states.reshape((-1, state_dim))
        #

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

        # Put back into original shape
        output_shape = list(orig_shape)
        output_shape[-1] = len(self.locations)
        output = output.reshape(output_shape)
        #
        return output


class BeaconMeasurementModel(dpf.MeasurementModel):

    def __init__(self, beacon_observer):
        super(BeaconMeasurementModel, self).__init__()
        self.beacon_observer = beacon_observer

    def forward(self, observations, states):
        assert(len(observations.shape) == 2)  # (N, num_beacons)
        assert(len(states.shape) == 3)  # (N, M, state_dim)

        # N := distinct trajectory count
        # M := particle count
        N, M, _ = states.shape
        _, num_beacons = observations.shape
        assert num_beacons == len(self.beacon_observer.locations)

        with torch.no_grad():
            log_likelihoods = torch.zeros((N, M)).to(observations.device)

            observations_pred = torch.from_numpy(
                self.beacon_observer(states.cpu().detach().numpy()).astype(np.float32)).to(observations.device)
            assert observations_pred.shape == (N, M, num_beacons)

            observations_error = observations_pred - observations[np.newaxis,:,:]
            assert observations_error.shape == (N, M, num_beacons)

            for i in range(num_beacons):
                variance = self.beacon_observer.variances[i]
                pdf = np.zeros((N,M))
                for j in range(N):
                    pdf[j] = scipy.stats.multivariate_normal.pdf(
                        observations_error[j,:,i].cpu().detach().numpy(), cov=variance)
                assert(pdf.shape == (N,M))
                log_likelihoods += torch.from_numpy(pdf.astype(np.float32)).to(observations.device)

        return log_likelihoods


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_out_channels, middle_channels=None, activation="leaky_relu"):
        super(ResidualBlock, self).__init__()

        if middle_channels == None:
            middle_channels = in_out_channels

        self.fc1 = nn.Linear(in_out_channels, middle_channels)
        self.fc2 = nn.Linear(middle_channels, in_out_channels)
        self.activation = activation_func(activation)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x

class DeepBeaconMeasurementModel(dpf.MeasurementModel):

    def __init__(self):
        super(DeepBeaconMeasurementModel, self).__init__()

        obs_dim = 3
        state_dim = 3

        Activation = nn.LeakyReLU
        units = 16
        self.observation_layers = nn.Sequential(
            nn.Linear(obs_dim, units // 2),
            # nn.Linear(units, units)
            # Activation(),
        )
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units // 2),
            # nn.Linear(units, units)
            # Activation(),
        )
        self.shared_layers = nn.Sequential(
            ResidualBlock(units),
            ResidualBlock(units),
            ResidualBlock(units),
            nn.Linear(units, 1),
            # nn.LogSigmoid()
        )

        self.units = units

    def forward(self, observations, states):
        assert(len(observations.shape) == 2)  # (N, obs_dim,)
        assert(len(states.shape) == 3)  # (N, M, state_dim)

        # N := distinct trajectory count
        # M := particle count

        N, M, _ = states.shape

        # (N, obs_dim) => (N, units // 2)
        observation_features = self.observation_layers(observations)
        assert observation_features.shape == (N, self.units // 2)

        # (N, units // 2) => (N, M, units // 2)
        observation_features = observation_features[:,np.newaxis,:].expand(N, M, self.units // 2)
        assert observation_features.shape == (N, M, self.units // 2)

        # (N, M, state_dim) => (N, M, units // 2)
        state_features = self.state_layers(states)
        assert state_features.shape == (N, M, self.units // 2)

        # (N, M, units)
        merged_features = torch.cat(
            (observation_features, state_features),
            dim=2)
        assert merged_features.shape == (N, M, self.units)

        # (N, M, units * 2) => (N, M, 1)
        log_likelihoods = self.shared_layers(merged_features)
        assert log_likelihoods.shape == (N, M, 1)

        # Return (N, M)
        return torch.squeeze(log_likelihoods, dim=2)

class DeepRobotDynamicsModel(dpf.DynamicsModel):

    def __init__(self):
        super(DeepRobotDynamicsModel, self).__init__()

        state_dim = 3
        control_dim = 2

        Activation = nn.LeakyReLU
        units = 16
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units // 2),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units // 2),
        )
        self.shared_layers = nn.Sequential(
            ResidualBlock(units),
            ResidualBlock(units),
            ResidualBlock(units),
            nn.Linear(units, state_dim),
            nn.Tanh()
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
        control_features = control_features[:,np.newaxis,:].expand(N, M, self.units // 2)
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
            dist = torch.distributions.Normal(torch.tensor([0.]), torch.tensor([0.2, 0.2, 0.1]))
            noise = dist.sample((N, M)).to(states_new.device)
            assert noise.shape == (N, M, state_dim)
            states_new = states_new + noise

        # Return (N, M, state_dim)
        return states_new


class RobotDynamicsModel(dpf.DynamicsModel):

    def forward(self, states_prev, controls, noisy=False):
        # states_prev:  (N, M, state_dim)
        # controls: (N, control_dim)
        #
        # N == distinct trajectory count
        # M == particle count

        N, M, state_dim = states_prev.shape

        v = controls[:, 0]
        omega = controls[:, 1]
        assert v.shape == (N,)
        assert omega.shape == (N,)

        if noisy:
            v_noise = np.random.normal(0, 0.1, size=(N,M)).astype(np.float32)
            v = v[:,np.newaxis] + torch.from_numpy(v_noise).to(v.device)

            omega_noise = np.random.normal(0, 0.05, size=(N,M)).astype(np.float32)
            omega = omega[:,np.newaxis] + torch.from_numpy(omega_noise).to(omega.device)

        states = torch.zeros_like(states_prev)
        states[:,:,0] = states_prev[:,:,0] + v * torch.cos(states_prev[:,:,2])
        states[:,:,1] = states_prev[:,:,1] + v * torch.sin(states_prev[:,:,2])
        states[:,:,2] = (states_prev[:,:,2] + omega) % (2 * np.pi)
        return states
