"""Check that the independent DoorKey predicates are control-sufficient."""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from envs.minigrid import MiniGrid


LEFT, RIGHT, FWD, PICKUP, TOGGLE = 0, 1, 2, 3, 4


def scripted(g):
    if g["agent_on_door"] and g["door_open"]:
        return FWD
    if not g["has_key"]:
        target = "key"
    elif g["door_between_agent_goal"]:
        target = "door"
    else:
        target = "goal"

    if g[f"{target}_reach"]:
        if target == "key":
            return PICKUP
        if target == "door" and not g["door_open"]:
            return TOGGLE
        return FWD
    if g[f"{target}_ahead"] and not g["wall_ahead"]:
        return FWD
    if g[f"{target}_left"]:
        return LEFT
    if g[f"{target}_right"]:
        return RIGHT
    return LEFT


def evaluate(task="doorkey6x6_rel", episodes=100, max_steps=300):
    successes, lengths = [], []
    for seed in range(episodes):
        env = MiniGrid(task, mode="eval", seed=seed, max_steps=max_steps)
        env.reset()
        success = False
        for step in range(max_steps):
            _, reward, done, _ = env.step(scripted(env.god_state()))
            if done:
                success = bool(reward > 0)
                break
        env.close()
        successes.append(success)
        lengths.append(step + 1)
    return float(np.mean(successes)), float(np.mean(lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="doorkey6x6_rel")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    success, length = evaluate(args.task, args.episodes, args.max_steps)
    print(f"{args.task}: success={success:.3f} ({round(success * args.episodes)}/{args.episodes}) mean_length={length:.1f}")


if __name__ == "__main__":
    main()
