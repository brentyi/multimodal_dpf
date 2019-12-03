import abc
import torch
import numpy as np

from . import panda_baseline_models
from . import panda_models
from . import dpf
from .utils import torch_utils
from .utils import misc_utils


class StateEstimator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args):
        pass

    def update(self, obs):
        # Pre-process observations
        observations = {}
        observations['gripper_pose'] = np.concatenate((
            obs['eef_pos'],
            obs['eef_quat'],
        ), axis=0)[np.newaxis, :]
        observations['gripper_sensors'] = np.concatenate((
            obs['contact-obs'][np.newaxis],
            obs['ee-force-obs'],
            obs['ee-torque-obs'],
        ), axis=0)[np.newaxis, :]
        observations['image'] = obs['image'][np.newaxis,:,:]

        control_keys = [
            'eef_pos',
            'eef_quat',
            'eef_vlin',
            'eef_vang',
            'ee-force-obs',
            'ee-torque-obs',
            'contact-obs'
        ]
        controls = []
        for key in control_keys:
            control = np.asarray(obs[key])
            if len(control.shape) == 0:
                control = control[np.newaxis]
            assert len(control.shape) == 1
            controls.append(control.astype(np.float32))
        controls = np.concatenate(controls, axis=0)[np.newaxis, :]

        return self._update(observations, controls)

    def _update(self, observations, controls):
        raise NotImplementedError


class GroundTruthStateEstimator(StateEstimator):
    def __init__(self):
        pass

    def update(self, obs):
        # Directly return door state
        return obs['object-state'][1]


class BaselineStateEstimator(StateEstimator):
    def __init__(self, experiment_name, initial_obs):

        # Create model
        self.model = panda_baseline_models.PandaBaselineModel(
            use_prev_state=True, units=32)
        self.prev_estimate = initial_obs['object-state'][1]

        # Create a buddy, who'll automatically load the latest checkpoint etc
        self.buddy = torch_utils.TrainingBuddy(
            experiment_name,
            self.model,
            log_dir="logs/baseline",
            checkpoint_dir="checkpoints/baseline")

    def _update(self, observations, controls):
        # Pre-process model inputs
        states_prev = np.array(self.prev_estimate)[np.newaxis, np.newaxis]

        # Prediction
        with torch.no_grad():
            states_new = self.model(*torch_utils.to_torch([
                states_prev,
                observations,
                controls
            ], device=self.buddy._device))

        # Post-process & return
        estimate = np.squeeze(states_new)
        self.prev_estimate = estimate
        return torch_utils.to_numpy(estimate)

class DPFStateEstimator(StateEstimator):
    def __init__(self, experiment_name, initial_obs):
        door_pos = initial_obs['object-state'][1]

        # Initialize models
        dynamics_model = panda_models.PandaSimpleDynamicsModel(state_noise=(0.05))
        measurement_model = panda_models.PandaMeasurementModel(units=32)
        self.pf_model = dpf.ParticleFilterNetwork(dynamics_model, measurement_model)

        # Create a buddy, who'll automatically load the latest checkpoint, etc
        self.buddy = torch_utils.TrainingBuddy(
            experiment_name,
            self.pf_model,
            optimizer_names=["e2e", "dynamics", "measurement"],
            log_dir="logs/pf",
            checkpoint_dir="checkpoints/pf"
        )

        # Initialize particles
        M = 200
        particles = np.zeros((1, M, 1))
        particles[:] = door_pos
        particles = torch_utils.to_torch(particles, device=self.buddy._device)
        log_weights = torch.ones((1, M), device=self.buddy._device) * (-np.log(M))

        self.particles = particles
        self.log_weights = log_weights

    def _update(self, observations, controls):
        # Run model
        state_estimates, new_particles, new_log_weights = self.pf_model.forward(
            self.particles,
            self.log_weights,
            *torch_utils.to_torch([
                observations,
                controls,
            ], device=self.buddy._device),
            resample=True,
            noisy_dynamics=True
        )

        self.particles = new_particles
        self.log_weights = new_log_weights

        return np.squeeze(torch_utils.to_numpy(state_estimates))
