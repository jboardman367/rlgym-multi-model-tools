import numpy as np
from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultiDiscreteWrapper, SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan, VecMonitor

from multi_model_tools import multi_learn

# DECLARE THE MODEL MAP HERE SO REWARD CAN ACCESS
model_map = [0, 0, 1, 2, 3, 3, 2, 0] # map of model indexes to players, should be of length = n_envs * players_per_env
learning_mask = [True, False, True, True] # learning mask is the same size as the models list

def normalise(vec:np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec

class SplitReward(DefaultReward):
    def __init__(self, number):
        self.index = number # this is to know what number RocketLeague instance it is in
        super().__init__()

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # look up which model it is
        model_num = model_map[self.index*len(state.players) + [bot.car_id for bot in state.players].index(player.car_id)]
        if model_num == 0:
            return -np.linalg.norm(player.car_data.angular_velocity) / 100 # this one will train to not flip
        elif model_num == 1:
            return 0 # we will be masking this one to not learn anyway
        elif model_num == 2:
            return (np.linalg.norm(player.car_data.position) - 2_000) / 5_000 # this one will train to go to the edges
        elif model_num == 3:
            to_ball = normalise(state.ball.position - player.car_data.position)
            return sum(to_ball[n] * player.car_data.linear_velocity[n] for n in range(3)) / 2000

# make a simple little class to send the indexes out to the reward functions
class CounterUpper:
    def __init__(self):
        self.value = -1

    def __call__(self):
        self.value +=1
        return self.value


if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    horizon = 2 * round(1 / (1 - gamma))  # Inspired by OpenAI Five
    print(f"fps={fps}, gamma={gamma}, horizon={horizon}")

    reward_indexer = CounterUpper()
    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=2,  # 2v2 for this example because why not
            tick_skip=frame_skip,
            reward_function=SplitReward(reward_indexer()),  # Simple reward since example code
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState()
        )


    rl_path = None  # Path to Epic installation (None so it uses login tricks)

    env = SB3MultipleInstanceEnv(rl_path, get_match, 2)     # Start 2 instances
    env = SB3MultiDiscreteWrapper(env)                      # Convert action space to multidiscrete
    env = VecCheckNan(env)                                  # Optional
    env = VecMonitor(env)                                   # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    models = []
    for _ in range(4):
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
    callbacks = [CheckpointCallback(round(1_000_000 / env.num_envs), save_path="policy", name_prefix=f"multi_{n}") for n in range(4)]

    # model.learn(100_000_000, callback=callback)


    multi_learn(
        models= models, # the list of models that will be used
        total_timesteps= 10_000_000_000, # total timestamps that will be trained for
        env= env,
        callbacks= callbacks, # list of callbacks, one for each model in the list of models
        num_players= 8, # team_size * num_instances
        model_map= model_map, # mapping of models to players. If this is also known to the reward function,
        #                             # one could allow each model to use a different reward
        learning_mask= learning_mask
    )

    # Now, if one wants to load a trained model from a checkpoint, use this function
    # This will contain all the attributes of the original model
    # Any attribute can be overwritten by using the custom_objects parameter,
    # which includes n_envs (number of agents), which has to be overwritten to use a different amount
    model = PPO.load("policy/rl_model_1000002_steps.zip", env, custom_objects=dict(n_envs=1))
    # Use reset_num_timesteps=False to keep going with same logger/checkpoints
    # model.learn(100_000_000, callback=callback, reset_num_timesteps=False)













