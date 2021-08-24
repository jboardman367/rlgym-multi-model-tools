import time
from collections import deque
from typing import List, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from time import perf_counter

# This function is heavily based off the collect_rollouts() method of the sb3 OnPolicyAlgorithm
def multi_collect_rollouts(
        env: VecEnv, models: List[PPO], model_map: list, all_last_obs: list,  n_rollout_steps: int,
        obs_size: int, all_callbacks: List[BaseCallback], learning_mask: List[bool]):
    p = perf_counter()
    n_steps = 0
    all_last_episode_restarts = [models[model_map[num]]._last_episode_starts for num in range(len(model_map))]
    for model in models:
        model.rollout_buffer.reset()

    for callback in all_callbacks: callback.on_rollout_start()

    models_length = len(models)
    map_length = len(model_map)
    print('setup: ', perf_counter() - p)
    p = perf_counter()
    while n_steps < n_rollout_steps:
        all_actions = []
        all_values = []
        all_log_probs = []
        all_clipped_actions = []
        pp = perf_counter()
        for obs_index in range(map_length):
            with th.no_grad():
                obs_tensor = obs_as_tensor(all_last_obs[obs_index], models[model_map[obs_index]].device)
                actions, values, log_probs = models[model_map[obs_index]].policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            clipped_actions = actions[0] # it is inside an extra layer for some reason, so take it out
            if isinstance(models[model_map[obs_index]], gym.spaces.Box):
                clipped_actions = np.clip(
                    actions,
                    models[model_map[obs_index]].action_space.low,
                    models[model_map[obs_index]].action_space.high
                )

            all_clipped_actions.append(clipped_actions)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
        print('getting model predictions: ', perf_counter() - pp)
        pp = perf_counter()
        flat_clipped_actions = np.array(all_clipped_actions)
        flat_new_obs, flat_rewards, flat_dones, flat_infos = env.step(flat_clipped_actions)
        infos_length = len(flat_infos) // 6
        all_infos = [flat_infos[x*infos_length:(x+1)*infos_length] for x in range(map_length)]
        all_rewards = [flat_rewards[x] for x in range(map_length)]
        print('getting env stuff: ', perf_counter() - pp)
        pp = perf_counter()

        for obs_index in range(map_length):
            models[model_map[obs_index]].num_timesteps += 1

        for callback in all_callbacks: callback.update_locals(locals())
        if any(callback.on_step() is False for callback in all_callbacks):
            return False

        pp = perf_counter()
        for model_index in range(models_length):
            models[model_index]._update_info_buffer(
                [all_infos[num][0] for num in range(map_length) if model_map[num] == model_index]
            ) # this should put the needed infos for each model in
        n_steps += 1
        print('putting shit into the info buffer: ', perf_counter() - pp)

        for obs_index in range(map_length):
            if isinstance(models[model_map[obs_index]].action_space, gym.spaces.Discrete):
                all_actions[obs_index] = all_actions[obs_index].reshape(-1,1)
        pp = perf_counter()
        for model_index in range(models_length):
            if learning_mask[model_index]: # skip learing where not necessary
                models[model_index].rollout_buffer.add( # disgusting list comprehension to send all the info to the buffer
                    np.asarray([all_last_obs[num][0] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_actions[num][0] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_rewards[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    np.asarray([all_last_episode_restarts[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    th.tensor([all_values[num] for num in range(len(model_map)) if model_map[num] == model_index]),
                    th.tensor([all_log_probs[num] for num in range(len(model_map)) if model_map[num] == model_index])
                )
        print("putting shit in the rollout buffer: ", perf_counter() - pp)

        new_obs_len = len(flat_new_obs) // map_length
        all_last_obs = [flat_new_obs[obs_index * new_obs_len:(obs_index + 1) * new_obs_len] for obs_index in range(map_length)]
        all_last_episode_restarts = flat_dones

    all_last_values, all_last_dones = [], []
    for obs_index in range(len(model_map)):
        with th.no_grad():
            # compute value for the last timestamp
            # the og code uses new_obs where I have last_obs, so I hope this still works since they should hold the same value
            obs_tensor = obs_as_tensor(all_last_obs[obs_index], models[model_map[obs_index]].device)
            _, values, _ = models[model_map[obs_index]].policy.forward(obs_tensor)
            all_last_values.append(values)

    for model_index in range(len(models)):
        models[model_index].rollout_buffer.compute_returns_and_advantage(
            last_values=th.tensor([all_last_values[num] for num in range(len(model_map)) if model_map[num] == model_index]),
            dones=np.asarray([all_last_episode_restarts[num] for num in range(len(model_map)) if model_map[num] == model_index])
        )

    for callback in all_callbacks: callback.on_rollout_end()
    return True

# This function is heavily based off the learn() method of the sb3 OnPolicyAlgorithm
def multi_learn(
        models: List[PPO],
        total_timesteps: int,
        env,
        num_players: int,
        learning_mask: Optional[List[bool]] = None,
        model_map: Optional[list] = None,
        callbacks: List[MaybeCallback] = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
):

    model_map = model_map or [n % len(models) for n in range(num_players)]
    learning_mask = learning_mask or [True for _ in range(len(models))]

    # make sure everything lines up
    assert len(models) == len(callbacks) == len(learning_mask)

    iteration = 0
    # this for loop is essentially the setup method, done for each model
    obs_size = len(env.reset()) // len(model_map)  # calculate the length of the each observation
    all_total_timesteps = []
    for model_index in range(len(models)):
        models[model_index].start_time = time.time()
        if models[model_index].ep_info_buffer is None or reset_num_timesteps:
            models[model_index].ep_info_buffer = deque(maxlen=100)
            models[model_index].ep_success_buffer = deque(maxlen=100)

        if models[model_index].action_noise is not None:
            models[model_index].action_noise.reset()

        if reset_num_timesteps:
            models[model_index].num_timesteps = 0
            models[model_index]._episode_num = 0
        else:
            # make sure training timestamps are ahead of internal counter
            total_timesteps += models[model_index].num_timesteps
        models[model_index]._total_timesteps = total_timesteps

        # leaving out the environment reset, since that will be done for all at once

        if eval_env is not None and models[model_index].seed is not None:
            eval_env.seed(models[model_index].seed)

        eval_env = models[model_index]._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not models[model_index]._custom_logger:
            models[model_index]._logger = utils.configure_logger(
                models[model_index].verbose, models[model_index].tensorboard_log, tb_log_name, reset_num_timesteps)

        callbacks[model_index] = models[model_index]._init_callback(
            callbacks[model_index], eval_env, eval_freq, n_eval_episodes, log_path=None)

        # instead of returning, I'm just going to shove these in lists
        all_total_timesteps.append(total_timesteps)

    for callback in callbacks: callback.on_training_start(locals(), globals())
    flat_last_obs = env.reset()
    all_last_obs = [flat_last_obs[x*obs_size:(x+1)*obs_size] for x in range(num_players)]

    # make sure the n_envs is correct for the models
    for model_index in range(len(models)):
        models[model_index].n_envs = model_map.count(model_index)
        models[model_index].rollout_buffer.n_envs = model_map.count(model_index)

    # I assume the correct thing here is to check each model separately for the while condition
    while all([models[i].num_timesteps < all_total_timesteps[i] for i in range(len(models))]):
        continue_training = multi_collect_rollouts(
            env, models, model_map, all_last_obs, min(model.n_steps for model in models), obs_size, callbacks, learning_mask
        )

        if continue_training is False:
            break

        iteration += 1
        for model in models:
            model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        # this is where the training info would be displayed
        for model_index in range(len(models)):
            if log_interval is not None and iteration % log_interval == 0 and learning_mask[model_index]:
                fps = int(models[model_index].num_timesteps / (time.time() - models[model_index].start_time))
                models[model_index].logger.record("time/iterations", iteration * model_map.count(model_index), exclude="tensorboard")
                if len(models[model_index].ep_info_buffer) > 0 and len(models[model_index].ep_info_buffer[0]) > 0:
                    models[model_index].logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in models[model_index].ep_info_buffer]))
                    models[model_index].logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in models[model_index].ep_info_buffer]))
                models[model_index].logger.record("time/fps", fps)
                models[model_index].logger.record("time/time_elapsed", int(time.time() - models[model_index].start_time), exclude="tensorboard")
                models[model_index].logger.record("time/total_timesteps", models[model_index].num_timesteps, exclude="tensorboard")
                models[model_index].logger.dump(step=models[model_index].num_timesteps)

        for model in models: model.train()


    for callback in callbacks: callback.on_training_end()

    return models