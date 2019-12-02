"""
Some helpers for moving along randomized waypoints
"""

import enum
import abc
import numpy as np

import robosuite

class AbstractWaypointPolicy(abc.ABC):

    class States(enum.Enum):
        READY = 1
        ACTIVE = 2
        WAITING = 3

    def __init__(self):
        self.switch_ready()

        self.counter = 0
        self.counter_threshold = 0

    def update(self, env):
        self.env = env
        if self._state == self.States.READY:
            return self._ready()
        elif self._state == self.States.ACTIVE:
            return self._active()
        elif self._state == self.States.WAITING:
            return self._waiting()
        else:
            assert False

    # Functions to override
    @abc.abstractmethod
    def get_initial_state(self):
        pass

    @abc.abstractmethod
    def _sample_waypoint(self):
        pass

    # Helpers for switching states

    def switch_ready(self):
        self._state = self.States.READY

    def switch_active(self):
        self._state = self.States.ACTIVE

    def switch_waiting(self):
        self._state = self.States.WAITING

    # State bodies

    def _ready(self):
        # Ready to start doing stuff -- let's sample a waypoint
        self.target_pos = self._sample_waypoint()
        self.delta_scale = np.exp(np.random.uniform(0., np.log(100)))

        self.counter = 0
        self.counter_threshold = np.random.uniform(100, 250)

        self.switch_active()

        action = np.array([0., 0., 0., -1])
        return action

    def _active(self):
        # Move toward waypoint
        hand_id = self.env.sim.model.body_name2id("right_hand")
        current_pos = self.env.sim.data.body_xpos[hand_id]
        dpos = self.target_pos - current_pos

        self.counter += 1
        if np.linalg.norm(dpos) <= 0.06 or self.counter >= self.counter_threshold or (
                self.counter >= 5 and np.linalg.norm(self.env._right_hand_total_velocity) < 0.01):
            self.switch_waiting()
            self.counter = 0
            self.counter_threshold = np.random.uniform(0, 5)
            self.delta_scale = 0.

        dpos *= self.delta_scale

        action = np.concatenate([dpos, [-1]])
        return action

    def _waiting(self):
        # Pause a lil before our next waypoint
        self.counter += 1
        if self.counter >= self.counter_threshold:
            self.switch_ready()

        action = np.array([0., 0., 0., -1])
        return action


class PushWaypointPolicy(AbstractWaypointPolicy):

    class PushStates(enum.Enum):
        RETRACTED = 1
        NEED_RETRACT = 2

    def __init__(self):
        super().__init__()
        self.push_state = self.PushStates.NEED_RETRACT
        self.push_x = 0.14

    def get_initial_state(self):
        initial_joints = np.array(
            [-0.055, -0.173, -0.983, -1.899, 1.48, 2.156, -1.125])
        initial_door = np.random.uniform(0.8, 1.5)
        return initial_joints, initial_door

    def _sample_waypoint(self):
        # Ready to start doing stuff -- let's sample a waypoint
        if self.push_state == self.PushStates.NEED_RETRACT:
            print("Retracting")
            waypoint = np.random.uniform(
                [0.14, -0.3, 1.544],
                [(0.14 + self.push_x) / 2., 0.19, 1.546],
            )
            self.push_state = self.PushStates.RETRACTED
        elif self.push_state == self.PushStates.RETRACTED:
            print("Pushing")
            waypoint = np.random.uniform(
                [self.push_x, -0.3, 1.544],
                [0.67, 0.19, 1.546],
            )
            self.push_x = waypoint[0]
            self.push_state = self.PushStates.NEED_RETRACT
        else:
            assert False

        return waypoint


class PullWaypointPolicy(AbstractWaypointPolicy):
    # Waypoints for pulling motion
    pull_waypoints = np.array([
        [0.582, 0.162, 1.546],
        [0.472, 0.122, 1.546],
        [0.339, -0.012, 1.546],
        [0.247, -0.219, 1.545],
        [0.283, -0.461, 1.545],
        [0.312, -0.549, 1.544],
    ])

    def get_initial_state(self):
        initial_joints = np.array(
            [-1.609, -0.615, 1.696, -1.627, 1.782, 3.228, -0.498])
        initial_door = 0.
        return initial_joints, initial_door

    def _sample_waypoint(self):
        alpha = np.random.uniform(0., 1.)
        waypoint = self._interpolate_waypoint(
            self.pull_waypoints, alpha)
        noise = np.random.normal(scale=(0.015, 0.015, 0.))
        return waypoint + noise

    def _interpolate_waypoint(self, waypoints, alpha):
        assert alpha >= 0. and alpha <= 1.

        if alpha <= 1e-9:
            return waypoints[0]

        distances = np.linalg.norm(waypoints[:-1] - waypoints[1:], axis=1)
        assert distances.shape == (len(waypoints) - 1,)
        cum_distances = np.zeros(len(waypoints))
        cum_distances[1:] = np.cumsum(distances)
        cum_distances /= cum_distances[-1]
        assert cum_distances.shape == (len(waypoints),)

        end_index = np.searchsorted(cum_distances, alpha)
        start_index = end_index - 1
        assert start_index >= 0

        offset = cum_distances[start_index]
        scale = cum_distances[end_index] - cum_distances[start_index]
        alpha = (alpha - offset) / scale
        assert alpha >= 0. and alpha <= 1.

        waypoint = (1 - alpha) * \
            waypoints[start_index] + alpha * waypoints[end_index]

        return waypoint
