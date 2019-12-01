"""
Helper for logging observations & writing them to an hdf5 file.
"""

import numpy as np
import h5py


def load_trajectories(*paths, use_vision=True,
                      vision_interval=10, use_proprioception=True, **unused):
    trajectories = []

    for path in paths:
        with TrajectoriesFile(path) as f:
            # Iterate over each trajectory
            for trajectory in f:
                # Possible keys:
                #   'joint_pos'
                #   'joint_vel'
                #   'gripper_qpos'
                #   'gripper_qvel'
                #   'eef_pos'
                #   'eef_quat'
                #   'eef_vlin'
                #   'eef_vang'
                #   'robot-state'
                #   'prev-act'
                #   'contact-obs'
                #   'ee-force-obs'
                #   'ee-torque-obs'
                #   'object-state'
                #   'image'

                # Pull out trajectory states -- this is just door position &
                # velocity
                states = trajectory['object-state'][:, 1:3]

                # Pull out observation states
                observations = {}
                observations['gripper_pose'] = np.concatenate((
                    trajectory['eef_pos'],
                    trajectory['eef_quat'],
                ), axis=1)
                observations['gripper_sensors'] = np.concatenate((
                    trajectory['contact-obs'][:, np.newaxis],
                    trajectory['ee-force-obs'],
                    trajectory['ee-torque-obs'],
                ), axis=1)
                if not use_proprioception:
                    observations['gripper_pose'][:] = 0
                    observations['gripper_sensors'][:] = 0

                observations['image'] = np.zeros_like(trajectory['image'])
                if use_vision:
                    observations['image'][::vision_interval] = trajectory['image'][::vision_interval]

                # Pull out control states
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
                    control = trajectory[key]
                    if len(control.shape) == 1:
                        control = control[:, np.newaxis]
                    assert len(control.shape) == 2
                    controls.append(control)
                controls = np.concatenate(controls, axis=1)

                if not use_proprioception:
                    controls[:] = 0

                timesteps = len(states)
                assert len(controls) == timesteps
                assert len(observations['image']) == timesteps

                trajectories.append((states, observations, controls))

    return trajectories


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
