"""
Helper for logging observations & writing them to an hdf5 file.
"""

import numpy as np
import h5py


class ObservationRecorder:
    def __init__(self, path, single_precision_floats=True):
        # Meta
        self.path = path
        self.single_precision_floats = single_precision_floats

        # Maps observation key => observation list
        self.obs_dict = {}

        # Count the number of trajectories that already exist
        self.trajectory_prefix = "trajectory"
        with self._h5py_file() as f:
            if len(f.keys()) > 0:
                prefix_length = len(self.trajectory_prefix)
                ids = [int(k[prefix_length:]) for k in f.keys()]
                self.trajectory_count = max(ids) + 1
            else:
                self.trajectory_count = 0
        assert type(self.trajectory_count) == int

    def push(self, obs):
        for key, value in obs.items():
            if key not in self.obs_dict:
                self.obs_dict[key] = []

            assert type(self.obs_dict[key]) == list
            self.obs_dict[key].append(np.copy(value))

    def save(self):
        with self._h5py_file() as f:
            # Put all pushed observations into a new group
            trajectory_name = self.trajectory_prefix + \
                str(self.trajectory_count)
            group = f.create_group(trajectory_name)
            for key, obs_list in self.obs_dict.items():
                # Convert list of observations to a numpy array
                data = np.array(obs_list)

                # Compress floats
                if data.type == np.float64 and self.single_precision_floats:
                    data = data.astype(np.float32)

                group.create_dataset(key, data=data, chunks=True)

        self.obs_dict = {}
        self.trajectory_count += 1

    def _h5py_file(self, mode='a'):
        return h5py.File(self.path, mode)
