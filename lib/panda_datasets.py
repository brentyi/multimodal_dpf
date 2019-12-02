import torch
import numpy as np

from . import dpf
from .utils import torch_utils
from .utils import misc_utils
from .utils import file_utils


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
                observation = misc_utils.DictIterator(observations)[t]
                control = controls[t]
                new_state = states[t]

                # Construct sample, bring to torch, & add to dataset
                sample = (prev_state, observation, control, new_state)
                sample = tuple(torch_utils.to_torch(x) for x in sample)

                if np.linalg.norm(new_state - prev_state) > 1e-5:
                    active_dataset.append(sample)
                else:
                    inactive_dataset.append(sample)

        print("Parsed data: {} active, {} inactive".format(
            len(active_dataset), len(inactive_dataset)))
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
    default_particle_variances = [0.1]
    default_subsequence_length = 20
    default_particle_count = 100

    def __init__(self, *paths, **kwargs):
        """
        Input:
          *paths: paths to dataset hdf5 files
        """

        trajectories = file_utils.load_trajectories(*paths, **kwargs)

        # Split up trajectories into subsequences
        super().__init__(trajectories, **kwargs)

        # Post-process subsequences; differentiate between active ones and inactive ones
        active_subsequences = []
        inactive_subsequences = []

        for subsequence in self.subsequences:
            start_state = subsequence[0][0]
            end_state = subsequence[0][-1]
            if np.linalg.norm(start_state - end_state) > 1e-5:
                active_subsequences.append(subsequence)
            else:
                inactive_subsequences.append(subsequence)

        print("Parsed data: {} active, {} inactive".format(
            len(active_subsequences), len(inactive_subsequences)))
        keep_count = min(
            len(active_subsequences) // 2,
            len(inactive_subsequences)
        )
        print("Keeping:", keep_count)

        np.random.shuffle(inactive_subsequences)
        self.subsequences = active_subsequences + \
            inactive_subsequences[:keep_count]
