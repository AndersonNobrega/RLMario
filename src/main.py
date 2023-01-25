import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import torch
# Gym is an OpenAI toolkit for RL
from gym.wrappers import FrameStack
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

from agent import Mario
from util import MetricLogger
from wrappers import GrayScaleObservation, ResizeObservation, SkipFrame

parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=RawTextHelpFormatter)

parser.add_argument('-l', '--load_model', action='store_true', help='')

args = vars(parser.parse_args())

print(args['load_model'])

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

checkpoint = Path('trained_mario.chkpt') if args['load_model'] else None

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

if args['load_model']:
    mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 50000
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # env.render()

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        if not args['load_model']:
            q, loss = mario.learn()
        else:
            q, loss = None, None

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)