import torch
import numpy as np

from . import dpf
from . import torch_utils
from . import file_utils


class PandaSimpleDataset(torch.utils.data.Dataset):
    """
    A customized data preprocessor for trajectories
    """

    def __init__(self, path, use_vision=True, use_vision_every=10,
                 use_proprioception=True, use_prev_states=True):
        """
        Input:
          path: path to dataset hdf5 file
        """

        trajectories = file_utils.load_trajectories(path)
        self.dataset = []
        for trajectory in trajectories:
            assert len(trajectory) == 3
            states, observations, controls = trajectory

            timesteps = len(states)
            assert type(observations) == dict
            assert len(controls) == timesteps

            for t in range(1, timesteps):
                # Pull out data & labels
                prev_state = states[t - 1]
                observation = torch_utils.DictIndexWrapper(observations)[t]
                if not use_prev_states:
                    prev_state[:] = 0
                if not use_vision or t % use_vision_every == 1:
                    observation['image'][:] = 0
                if not use_proprioception:
                    observation['gripper_pose'][:] = 0
                    observation['gripper_sensors'][:] = 0
                control = controls[t]
                new_state = states[t]

                # Construct sample, bring to torch, & add to dataset
                sample = (prev_state, observation, control, new_state)
                sample = tuple(torch_utils.to_torch(x) for x in sample)
                self.dataset.append(sample)

    def __getitem__(self, index):
        """ Get a subsequence from our dataset
        Output:
            sample: (prev_state, observation, control, new_state)
        """
        return self.dataset[index]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.dataset)


class PandaParticleFilterDataset(dpf.ParticleFilterDataset):
    pass
