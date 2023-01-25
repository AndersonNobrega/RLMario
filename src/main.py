import datetime
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

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


def get_args():
    parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-c', '--load_checkpoint', type=str, help='Path to load checkpoint file for training.')
    parser.add_argument('-r', '--replay', type=str, help='Path to load trained model to play.')

    return vars(parser.parse_args())


def create_game_env():
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

    return env


def train_agent(env, checkpoint_path):
    save_dir = pathlib.Path().resolve() / pathlib.Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint_path)

    logger = MetricLogger(save_dir)

    episodes = 40000
    for e in range(episodes):
        state = env.reset()

        # Play the game!
        while True:
            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

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


def replay(env, agent_path):
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, checkpoint=agent_path)
    mario.exploration_rate = mario.exploration_rate_min

    while True:
        state = env.reset()

        while True:
            env.render()

            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            state = next_state

            if done or info["flag_get"]:
                break


def main():
    # Get CLI args
    args = get_args()

    # Get game environment after post processing steps
    env = create_game_env()

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}\n")

    if args['replay'] is not None:
        replay(env, args['replay'])
    else:
        train_agent(env, args['load_checkpoint'])


if __name__ == "__main__":
    main()
