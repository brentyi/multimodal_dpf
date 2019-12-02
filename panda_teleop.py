import argparse
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
import robosuite.devices
from robosuite.wrappers import IKWrapper

if __name__ == "__main__":
    env = robosuite.make(
        "PandaDoor",
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=50,
        controller='position',
    )

    # enable controlling the end effector directly instead of using joint velocities
    # env = IKWrapper(env)

    # initialize device

    device = robosuite.devices.Keyboard()
    env.viewer.add_keypress_callback("any", device.on_press)
    env.viewer.add_keyup_callback("any", device.on_release)
    env.viewer.add_keyrepeat_callback("any", device.on_press)

    while True:
        obs = env.reset()
        env.controller.step = 0.
        env.controller.last_goal_position = np.array((0, 0, 0))
        env.controller.last_goal_orientation = np.eye(3)

        sim = env.sim
        hinge_id = sim.model.get_joint_qpos_addr("door_hinge")
        # sim.data.qpos[hinge_id] = 1.

        # env.viewer.set_camera(camera_id=1)
        env.render()

        # rotate the gripper so we can see it easily
        env.set_robot_joint_positions(
            [-1.609, -0.615, 1.696, -1.627, 1.782, 3.228, -0.498])
        # env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
        # env.set_robot_joint_positions([1.34262204e-03, 1.91835113e-02, 2.10174521e-01, - 1.73815323e+00, - 1.54009070e+00, 2.82023969e+00, 2.26093989e+00])

        device.start_control()

        while True:
            state = device.get_controller_state()
            dpos, rotation, grasp, reset = (
                state["dpos"],
                state["rotation"],
                state["grasp"],
                state["reset"],
            )
            if reset:
                break

            # map 0 to -1 (open) and 1 to 0 (closed halfway)
            grasp = grasp - 1

            action = np.concatenate([dpos * 500, [grasp]])
            obs, reward, done, info = env.step(action)
            env.render()

            sim = env.sim
            hinge_id = sim.model.get_joint_qpos_addr("door_hinge")

            print(">>>>")
            print("Door pos", sim.data.qpos[hinge_id])
            print("Hand pos", list(np.round(obs['eef_pos'], 3)))
            print("Joint pos", list(np.round(obs['joint_pos'], 3)))
            print("Force obs", np.round(obs['ee-force-obs'], 3))
            print("Torque obs", np.round(obs['ee-torque-obs'], 3))
            print("Contact obs", obs['contact-obs'])

            # 'eef_pos',
            # 'eef_pos',
            # 'eef_pos',
