### DEAD RECKONING TEST

# def eval(trajectories, max_timesteps=100000):
#     # To make things easier, we're going to cut all our trajectories to the same length :)
#     timesteps = np.min([len(s) for s, _, _ in trajectories] + [max_timesteps])
#     predicted_states = [[states[0]] for states, _, _ in trajectories]
#     actual_states = [states[:timesteps] for states, _, _ in trajectories]

#     for t in range(1, timesteps):
#         for i in range(len(trajectories)):
#             prev_state = predicted_states[i][t - 1]
#             new_vel = actual_states[i][t][1]
#             new_pos = prev_state[0] + new_vel / 20

#             predicted_states[i].append([new_pos, new_vel])
#         misc_utils.progress_bar(t / timesteps)
#     misc_utils.progress_bar(1.)

#     predicted_states = np.array(predicted_states)
#     actual_states = np.array(actual_states)
#     return predicted_states, actual_states


# eval_trajectories = file_utils.load_trajectories(
#     "data/pull-test-small.hdf5",
#     use_proprioception=True,
#     use_vision=True,
#     vision_interval=1
# )
# pred, actual = eval(eval_trajectories[0:1], max_timesteps=200)
# vis_eval(pred, actual)

