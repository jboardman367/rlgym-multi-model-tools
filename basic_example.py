import numpy as np
from rlgym.envs import Match
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultiDiscreteWrapper, SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan, VecMonitor

from multi_model_tools import multi_learn


def normalise(vec:np.ndarray):
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    horizon = 2 * round(1 / (1 - gamma))  # Inspired by OpenAI Five
    print(f"fps={fps}, gamma={gamma}, horizon={horizon}")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=3,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=DefaultReward(),  # Simple reward since example code
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState()
        )


    rl_path = None  # Path to Epic installation (None so it uses login tricks)

    env = SB3MultipleInstanceEnv(rl_path, get_match, 1)     # Start 2 instances, waiting 60 seconds between each
    env = SB3MultiDiscreteWrapper(env)                      # Convert action space to multidiscrete
    env = VecCheckNan(env)                                  # Optional
    env = VecMonitor(env)                                   # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    models = []
    for _ in range(6):
        model = PPO(
            'MlpPolicy',
            env,
            n_epochs=10,                 # PPO calls for multiple epochs, SB3 does early stopping to maintain target kl
            target_kl=0.02 / 1.5,        # KL to aim for (divided by 1.5 because it's multiplied later for unknown reasons)
            learning_rate=3e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=horizon,          # Batch size as high as possible within reason
            n_steps=horizon,             # Number of steps to perform before optimizing network
            tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )
        models.append(model)

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callbacks = [CheckpointCallback(round(1_000_000 / env.num_envs), save_path="policy", name_prefix=f"multi_{n}") for n in range(6)]

    # model.learn(100_000_000, callback=callback)
    last_obs = env.reset()
    model_map = list(range(6))
    obs_size = len(last_obs) // len(model_map)

    multi_learn(
        models= models,
        total_timesteps= 10_000_000_000,
        obs_size= obs_size,
        env= env,
        callbacks= callbacks
    )


    # Now, if one wants to load a trained model from a checkpoint, use this function
    # This will contain all the attributes of the original model
    # Any attribute can be overwritten by using the custom_objects parameter,
    # which includes n_envs (number of agents), which has to be overwritten to use a different amount
    model = PPO.load("policy/rl_model_1000002_steps.zip", env, custom_objects=dict(n_envs=1))
    # Use reset_num_timesteps=False to keep going with same logger/checkpoints
    # model.learn(100_000_000, callback=callback, reset_num_timesteps=False)













