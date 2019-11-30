from . import dpf
from . import torch_utils


class PandaBaselineDataset(torch.utils.data.Dataset):
    """
    A customized data preprocessor for trajectories
    """

    def __init__(self, trajectories):
        """
        Input:
          trajectories: list of trajectories, where each is a tuple of (states, observations, controls)
        """

        self.dataset = []
        for trajectory in trajectories:
            assert len(trajectory) == 3
            states, observations, controls = trajectory

            timesteps = len(states)
            assert len(observations) == timesteps
            assert len(controls) == timesteps

            for t in range(1, timesteps):
                # Pull out data & labels
                prev_state = states[t - 1]
                observation = observations[t]
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
