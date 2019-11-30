import torch
import numpy as np

from . import dpf
from . import torch_utils
from . import file_utils


class PandaSimpleDataset(torch.utils.data.Dataset):
    """
    A customized data preprocessor for trajectories
    """

    def __init__(self, *paths, **kwargs):
        """
        Input:
          *paths: paths to dataset hdf5 files
        """

        trajectories = file_utils.load_trajectories(*paths, **kwargs)
        active_dataset = []
        inactive_dataset = []
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
                control = controls[t]
                new_state = states[t]

                # Construct sample, bring to torch, & add to dataset
                sample = (prev_state, observation, control, new_state)
                sample = tuple(torch_utils.to_torch(x) for x in sample)

                if np.linalg.norm(new_state - prev_state) > 1e-5 and new_state[0] >= 1e-5:
                    active_dataset.append(sample)
                else:
                    inactive_dataset.append(sample)

        print("Parsed data: {} active, {} inactive".format(len(active_dataset), len(inactive_dataset)))
        keep_count = min(len(active_dataset) // 2, len(inactive_dataset))
        print("Keeping:", keep_count)
        np.random.shuffle(inactive_dataset)
        self.dataset = active_dataset + inactive_dataset[:keep_count]

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
