#!/usr/bin/env python3
# File: evaluate_rllib_hmarl.py
# Usage: python evaluate_rllib_hmarl.py --checkpoint /path/to/checkpoint --output_path /path/to/save --max-eps 100

import os
import sys
import json
import time
from datetime import datetime
from statistics import mean, stdev
from tqdm import tqdm
import argparse
import logging

# === Add project path if needed ===
CAGE_ROOT = "/projects/ImaniLab/Armita/CAGE-Multi-Adversary"
if CAGE_ROOT not in sys.path:
    sys.path.insert(0, CAGE_ROOT)

# reduce ray verbosity
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.WARN)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)

import ray
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOTorchPolicy

# Import your custom model, env wrapper and CybORG pieces
from hmarl_action_mask_model import TorchActionMaskModelHppo
from EnterpriseMAEHmarl import EnterpriseMAE
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
import gymnasium

# import your training config builder
from train_subpolicies import build_algo_config

# ----------------- Configurable constants -----------------
NUM_AGENTS = 5
EPISODE_LENGTH = 500

# Build same POLICY_MAP used during training
POLICY_MAP = {}
for i in range(NUM_AGENTS):
    POLICY_MAP[f"blue_agent_{i}_master"]  = f"Agent{i}_master"
    POLICY_MAP[f"blue_agent_{i}_investigate"]  = f"Agent{i}_investigate"
    POLICY_MAP[f"blue_agent_{i}_recover"]  = f"Agent{i}_recover"

def policy_mapper(agent_id, *_):
    return POLICY_MAP[agent_id]


 # import custom red agents
from FSRedAgentCombined import FSRedAgentCombined
from FSRedAgentDegrader import FSRedAgentDegrader
from FSRedAgentFailureTracker import FSRedAgentFailureTracker
from FSRedAgentImpacter import FSRedAgentImpacter
from FSRedAgentTargetA import FSRedAgentTargetA
from FSRedAgentTargetAOperational import FSRedAgentTargetAOperational
from FSRedAgentTargetARestricted import FSRedAgentTargetARestricted
from FSRedAgentTargetB import FSRedAgentTargetB
from FSRedAgentTargetBOperational import FSRedAgentTargetBOperational
from FSRedAgentTargetBRestricted import FSRedAgentTargetBRestricted


RED_AGENT_CLASSES = [
    FiniteStateRedAgent,
    FSRedAgentCombined,
    FSRedAgentDegrader,
    FSRedAgentFailureTracker,
    FSRedAgentImpacter,
    FSRedAgentTargetA,
    FSRedAgentTargetAOperational,
    FSRedAgentTargetARestricted,
    FSRedAgentTargetB,
    FSRedAgentTargetBOperational,
    FSRedAgentTargetBRestricted,
    SleepAgent,

]

# env creator (same as training)

# ----------------- Helper: recursive mkdir -----------------
def rmkdir(path: str):
    if path == "":
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ----------------- Main evaluation routine -----------------
def evaluate_manual(rac, checkpoint: str, output_path: str, max_eps: int =100, seed=None, runtime_env=None):
    # Start ray
    if not ray.is_initialized():
        if runtime_env:
            ray.init(runtime_env=runtime_env, include_dashboard=False)
        else:
            ray.init(include_dashboard=False)

    # Build algorithm config (same as training)
    config = build_algo_config()

    # To avoid RLlib creating env runners for eval, keep manual execution:
    # Build algorithm and restore
    print("before algo.restore")
    algo = config.build()
    algo.restore(checkpoint)
    print("after algo.resstore")

    # instantiate cyborg + wrapper explicitly so we can call cyborg.get_last_action(...)
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=rac,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped = EnterpriseMAE(env=cyborg)

    # Prepare outputs
    if not output_path.endswith("/"):
        output_path += "/"
    rmkdir(output_path)

    version_header = f"CybORG v{CYBORG_VERSION}, Scenario4"
    author_header = f"Evaluated checkpoint: {os.path.abspath(checkpoint)}"

    total_reward = []
    actions_log = []
    obs_log = []
    step_reward_log = []

    start = datetime.now()

    agents_list = list(wrapped.agents) if hasattr(wrapped, "agents") else []  # fallback

    for ep in tqdm(range(max_eps), desc="Episodes"):
        observations, _ = wrapped.reset()
        r = []
        a = []
        o = []
        for step in range(EPISODE_LENGTH):
            # Build actions dict by querying each agent's policy with the observation used at train time
            actions = {}
            for agent_name, obs in observations.items():
                # Only act for known blue agents mapped in POLICY_MAP
                if agent_name not in POLICY_MAP:
                    continue
                policy_id = policy_mapper(agent_name, None, None)
                policy = algo.get_policy(policy_id)
                # compute_single_action returns (action, state_out, info); we take [0]
                act = policy.compute_single_action(obs)[0]
                actions[agent_name] = act

            # Step env
            observations, rew, term, trunc, info = wrapped.step(actions)

            # stop if all done
            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped.agents
            } if hasattr(wrapped, "agents") else {}

            if wrapped.agents and all(done.values()):
                # fetch last actions from cyborg (env) before breaking
                # record step data
                last_actions = {
                    agent_name: cyborg.get_last_action(agent_name)
                    for agent_name in cyborg.agents
                    if agent_name in cyborg.agents
                } if hasattr(cyborg, "get_last_action") else {}
                a.append(last_actions)
                o.append({agent_name: observations.get(agent_name) for agent_name in observations.keys()})
                r.append(mean(rew.values()) if len(rew) > 0 else 0.0)
                break

            # record per-step mean reward across agents (baseline behavior)
            r.append(mean(rew.values()) if len(rew) > 0 else 0.0)

            # Save env-clean observations and cyborg-reported actions when requested
            last_actions = {
                agent_name: cyborg.get_last_action(agent_name)
                for agent_name in cyborg.agents
                if hasattr(cyborg, "get_last_action")
            } if hasattr(cyborg, "get_last_action") else {k: actions.get(k) for k in actions.keys()}

            a.append(last_actions)
            o.append({agent_name: observations.get(agent_name) for agent_name in observations.keys()})

        # episode done
        total_reward.append(sum(r))
        actions_log.append(a)
        obs_log.append(o)
        step_reward_log.append(r)

    end = datetime.now()
    elapsed = end - start

    # Stats
    reward_mean = mean(total_reward) if total_reward else 0.0
    reward_stdev = stdev(total_reward) if len(total_reward) > 1 else 0.0
    reward_string = f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"

    # Print (concise)
    print(version_header)
    print(author_header)
    print(reward_string)
    print(f"Evaluation took: {elapsed}")

    # Save files to match baseline structure
    with open(os.path.join(output_path, "summary.txt"), "w") as f:
        f.write(version_header + "\n")
        f.write(author_header + "\n")
        f.write(reward_string + "\n")
        f.write(f"checkpoint: {checkpoint}\n")

    with open(os.path.join(output_path, "full.txt"), "w") as f:
        f.write(version_header + "\n")
        f.write(author_header + "\n")
        f.write(reward_string + "\n")
        for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
            f.write(f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n")

    with open(os.path.join(output_path, "actions.txt"), "w") as f:
        f.write(version_header + "\n")
        f.write(author_header + "\n")
        f.write(reward_string + "\n")
        for act in actions_log:
            f.write(f"actions: {act}\n")

    summary_json = {
        "submission": {
            "author": "RLlib-HMARL-eval",
            "team": "N/A",
            "technique": "PPO + HMARL",
        },
        "parameters": {
            "seed": seed,
            "episode_length": EPISODE_LENGTH,
            "max_episodes": max_eps,
        },
        "time": {
            "start": str(start),
            "end": str(end),
            "elapsed": str(elapsed),
        },
        "reward": {
            "mean": reward_mean,
            "stdev": reward_stdev,
        },
        "agents": {
            # no direct textual representation for policies here
            "policy_map": POLICY_MAP
        },
        "checkpoint": os.path.abspath(checkpoint),
    }
    with open(os.path.join(output_path, "summary.json"), "w") as out:
        json.dump(summary_json, out, indent=2)

    with open(os.path.join(output_path, "scores.txt"), "w") as out:
        out.write(f"reward_mean: {reward_mean}\n")
        out.write(f"reward_stdev: {reward_stdev}\n")

    with open(os.path.join(output_path, "step_rewards.txt"), "w") as f:
        f.write(version_header + "\n")
        f.write(author_header + "\n")
        for ep_idx, ep_rewards in enumerate(step_reward_log):
            f.write(f"Episode {ep_idx} step_rewards: {ep_rewards}\n")

    # finish
    try:
        algo.stop()
    except Exception:
        pass
    ray.shutdown()

# ----------------- CLI -----------------
if __name__ == "__main__":
    for rac in RED_AGENT_CLASSES:
        # RAC = rac   
        red_agent_class = rac.__name__
        print("\n==============================")
        print(f" Evaluating Red Agent: {red_agent_class}")
        print("==============================\n")
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", type=str, default="/projects/ImaniLab/Armita/CAGE-Multi-Adversary/defense_baselines/HMARL/HMARLforFSRedOnly/PPO_CC4_4af0e_00000_0_2025-12-09_12-12-51/checkpoint_000199", help="RLlib checkpoint path")
        parser.add_argument("--output_path", type=str, default="/projects/ImaniLab/Armita/CAGE-Multi-Adversary/defense_baselines/HMARL/HMARLforFSRedOnly" + str(red_agent_class), help="Directory to save evaluation outputs")
        parser.add_argument("--max-eps", type=int, default=100, help="How many episodes to run")
        parser.add_argument("--seed", type=int, default=None)
        args = parser.parse_args()
    
        runtime_env = {
            "working_dir": os.path.join(CAGE_ROOT, "CybORG"),
            "env_vars": {"PYTHONPATH": CAGE_ROOT},
        }

        def env_creator_CC4(_):
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=rac,
                steps=EPISODE_LENGTH,
            )
            cyborg = CybORG(scenario_generator=sg)
            return EnterpriseMAE(env=cyborg)

            # Register env & custom model same as training
        register_env("CC4", env_creator_CC4)
        ModelCatalog.register_custom_model("hmarl_model", TorchActionMaskModelHppo)
    
        evaluate_manual(rac, checkpoint=args.checkpoint, output_path=args.output_path, max_eps=args.max_eps, seed=args.seed, runtime_env=runtime_env)
        