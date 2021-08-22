import time
from collections import deque
from typing import List, Optional

import gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

def multi_collect_rollouts(env: VecEnv, models: List[PPO],  n_rollout_steps: int, obs_size: int, all_callbacks: List[BaseCallback]):
    n_steps = 0
    for model in models:
        model.rollout_buffer.reset()

    for callback in all_callbacks: callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        all_actions = []
        all_values = []
        all_log_probs = []
        all_clipped_actions = []
        for i in range(6):
            with th.no_grad():
                obs_tensor = obs_as_tensor(models[i]._last_obs, models[i].device)
                actions, values, log_probs = models[i].policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            actions = actions[0] # it is inside an extra layer for some reason, so take it out
            clipped_actions = actions
            if isinstance(models[i], gym.spaces.Box):
                clipped_actions = np.clip(actions, models[i].action_space.low, models[i].action_space.high)
            all_clipped_actions.append(clipped_actions)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
        flat_clipped_actions = np.asarray(all_clipped_actions)
        flat_new_obs, flat_rewards, flat_dones, flat_infos = env.step(flat_clipped_actions)
        infos_length = len(flat_infos) // 6
        all_infos = [flat_infos[x*infos_length:(x+1)*infos_length] for x in range(6)]
        all_rewards = [flat_rewards[x] for x in range(6)]

        for model in models:
            model.num_timesteps += 1

        for callback in all_callbacks: callback.update_locals(locals())
        if all(callback.on_step() is False for callback in all_callbacks):
            return False

        for i in range(6):
            models[i]._update_info_buffer(all_infos[i])
        n_steps += 1

        for i in range(6):
            if isinstance(models[i].action_space, gym.spaces.Discrete):
                all_actions[i] = all_actions[i].reshape(-1,1)

            models[i].rollout_buffer.add(
                models[i]._last_obs[0], all_actions[i], all_rewards[i],
                models[i]._last_episode_starts, all_values[i], all_log_probs[i]
            )
            models[i]._last_obs = flat_new_obs[i*obs_size:(i+1)*obs_size]
            models[i]._last_episode_starts = flat_dones[i]

    for i in range(6):
        with th.no_grad():
            # compute value for the last timestamp
            # the og code uses new_obs where I have last_obs, so I hope this still works since they should hold the same value
            obs_tensor = obs_as_tensor(models[i]._last_obs, models[i].device)
            _, values, _ = models[i].policy.forward(obs_tensor)

        # this line also has a similar thing with dones instead of last_episode_starts in the og
        models[i].rollout_buffer.compute_returns_and_advantage(last_values=values, dones=models[i]._last_episode_starts)

    for callback in all_callbacks: callback.on_rollout_end()
    return True

def multi_learn(
        models: List[PPO],
        total_timesteps: int,
        obs_size: int,
        env,
        callbacks: List[MaybeCallback] = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
):
    iteration = 0
    # this for loop is essentially the setup method
    all_total_timesteps = []
    for i in range(6):
        models[i].start_time = time.time()
        if models[i].ep_info_buffer is None or reset_num_timesteps:
            models[i].ep_info_buffer = deque(maxlen=100)
            models[i].ep_success_buffer = deque(maxlen=100)

        if models[i].action_noise is not None:
            models[i].action_noise.reset()

        if reset_num_timesteps:
            models[i].num_timesteps = 0
            models[i]._episode_num = 0
        else:
            # make sure training timestamps are ahead of internal counter
            total_timesteps += models[i].num_timesteps
        models[i]._total_timesteps = total_timesteps

        # leaving out the environment reset, since that will be done for all at once

        if eval_env is not None and models[i].seed is not None:
            eval_env.seed(models[i].seed)

        eval_env = models[i]._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not models[i]._custom_logger:
            models[i]._logger = utils.configure_logger(models[i].verbose, models[i].tensorboard_log, tb_log_name, reset_num_timesteps)

        callbacks[i] = models[i]._init_callback(callbacks[i], eval_env, eval_freq, n_eval_episodes, log_path=None)

        # instead of returning, I'm just going to shove these in lists
        all_total_timesteps.append(total_timesteps)

    for callback in callbacks: callback.on_training_start(locals(), globals())
    flat_last_obs = env.reset()
    all_last_obs = [flat_last_obs[x*obs_size:(x+1)*obs_size] for x in range(6)]
    for i in range(6):
        models[i]._last_obs = all_last_obs[i]
        models[i]._last_episode_starts = np.ones(1, dtype=bool)

    # I assume the correct thing here is to check each model separately for the while condition
    while all([models[i].num_timesteps < all_total_timesteps[i] for i in range(6)]):
        continue_training = multi_collect_rollouts(
            env, models, min(model.n_steps for model in models), obs_size, callbacks
        )

        if continue_training is False:
            break

        iteration += 1
        for model in models:
            model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        # this is where the training info would be displayed
        for model in models:
            if log_interval is not None and iteration % log_interval == 0 :
                fps = int(model.num_timesteps / (time.time() - model.start_time))
                model.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
                    model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
                    model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))
                model.logger.record("time/fps", fps)
                model.logger.record("time/time_elapsed", int(time.time() - model.start_time), exclude="tensorboard")
                model.logger.record("time/total_timesteps", model.num_timesteps, exclude="tensorboard")
                model.logger.dump(step=model.num_timesteps)

        for model in models: model.train()


    for callback in callbacks: callback.on_training_end()

    return models