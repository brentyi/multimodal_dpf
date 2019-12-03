import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

from lib import panda_waypoint_policies
from lib import panda_state_estimators
from lib.utils import file_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--visualize_observations', action='store_true')
    args = parser.parse_args()

    ### <SETTINGS>
    preview_mode = args.preview
    vis_images = args.visualize_observations
    ### </SETTINGS>

    if preview_mode:
        vis_images = False

    env = robosuite.make(
        "PandaDoor",
        has_renderer=preview_mode,
        ignore_done=True,
        use_camera_obs=(not preview_mode),
        camera_name="birdview",
        camera_height=32,
        camera_width=32,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=20,
        controller='position',
    )

    errors = []
    for rollout_index in range(10):
        obs = env.reset()
        if preview_mode:
            env.render()
        env.controller.step = 0.
        env.controller.last_goal_position = np.array((0, 0, 0))
        env.controller.last_goal_orientation = np.eye(3)

        # Initialize movement policy
        # (note that we're not directly using the waypoint policy here -- see below)
        policy = panda_waypoint_policies.PullWaypointPolicy()

        # Initialize state estimator
        estimator = panda_state_estimators.GroundTruthStateEstimator()
        # estimator = panda_state_estimators.BaselineStateEstimator("baseline_all_sensors", obs)
        # estimator = panda_state_estimators.DPFStateEstimator("dpf_all_sensors", obs)

        # Set initial joint and door position
        initial_joints, initial_door = policy.get_initial_state()
        env.set_robot_joint_positions(initial_joints)
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(
            "door_hinge")] = initial_door

        target_door_pos = 1.
        waypoint_alpha = 0

        if vis_images:
            plt.figure()
            plt.gca().invert_yaxis()
            plt.ion()
            plt.show()

        max_iteration_count = 500
        for i in range(max_iteration_count):
            print("\r#{}:{}".format(rollout_index, i), end="")

            # Estimate door position & pass to P controller
            door_pos = estimator.update(obs)
            waypoint_alpha += (target_door_pos - door_pos) * 0.015
            waypoint_alpha = np.clip(waypoint_alpha, 0., 1.)

            eef_pos = obs['eef_pos']
            action = 100 * (policy._interpolate_waypoint(
                policy.pull_waypoints, waypoint_alpha) - eef_pos)
            action = np.append(action, -1)

            obs, reward, done, info = env.step(action)
            if preview_mode:
                env.render()

            if vis_images:
                start = time.time()
                plt.imshow(obs['image'], cmap='gray')
                plt.draw()
                plt.pause(0.0001)

        actual_door_pos = obs['object-state'][1]
        error = actual_door_pos - target_door_pos
        errors.append(error)

    print()
    print()
    print("MSE ERROR")
    print(np.mean(errors))
