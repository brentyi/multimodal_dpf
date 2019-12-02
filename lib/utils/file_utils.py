"""
Helper for logging observations & writing them to an hdf5 file.
"""

import numpy as np
import h5py


class TrajectoriesFile:
    def __init__(self, path, single_precision_floats=True, compress=True):
        assert path[-5:] == ".hdf5", "Missing file extension!"

        # Meta
        self._path = path
        self._single_precision_floats = single_precision_floats
        self._compress = compress

        # Maps observation key => observation list
        self._obs_dict = {}

        # Count the number of trajectories that already exist
        self._trajectory_prefix = "trajectory"
        with self._h5py_file() as f:
            if len(f.keys()) > 0:
                prefix_length = len(self._trajectory_prefix)
                ids = [int(k[prefix_length:]) for k in f.keys()]
                self._trajectory_count = max(ids) + 1
            else:
                self._trajectory_count = 0
        assert type(self._trajectory_count) == int

        self._file = None

    def __enter__(self):
        if self._file is None:
            self._file = self._h5py_file()
        return self

    def __getitem__(self, index):
        assert self._file is not None, "Not called in with statement!"

        # Check that the index is sane
        assert type(index) == int
        if index >= len(self):
            # For use as a standard Python iterator
            raise IndexError

        traj_key = self._trajectory_prefix + str(index)
        assert traj_key in self._file.keys()

        # Copy values to numpy array
        output = {}
        for key, value in self._file[traj_key].items():
            output[key] = value[:]
            assert type(output[key]) == np.ndarray

        return output

    def __len__(self):
        return self._trajectory_count

    def __exit__(self, *unused):
        # Close the file
        if self._file is not None:
            self._file.close()
            self._file = None

    def add_timestep(self, obs):
        for key, value in obs.items():
            if key not in self._obs_dict:
                self._obs_dict[key] = []

            assert type(self._obs_dict[key]) == list
            self._obs_dict[key].append(np.copy(value))

    def end_trajectory(self):
        assert self._file is not None, "Not called in with statement!"

        # Put all pushed observations into a new group
        trajectory_name = self._trajectory_prefix + \
            str(self._trajectory_count)
        group = self._file.create_group(trajectory_name)
        for key, obs_list in self._obs_dict.items():
            # Convert list of observations to a numpy array
            data = np.array(obs_list)

            # Compress floats
            if data.dtype == np.float64 and self._single_precision_floats:
                data = data.astype(np.float32)

            if self._compress:
                group.create_dataset(
                    key, data=data, chunks=True, compression="gzip")
            else:
                group.create_dataset(key, data=data, chunks=True)

        self._obs_dict = {}
        self._trajectory_count += 1

    def reencode(self, target_path):
        source = self._h5py_file()
        target = TrajectoriesFile(target_path)
        with source, target:
            for name, trajectory in source.items():
                keys = trajectory.keys()
                for obs_step in zip(*trajectory.values()):
                    target.add_timestep(dict(zip(keys, obs_step)))
                target.end_trajectory()
                print("Wrote ", name)
        return target

    def _h5py_file(self, mode='a'):
        return h5py.File(self._path, mode)