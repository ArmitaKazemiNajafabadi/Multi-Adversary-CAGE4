import inspect
import time
from statistics import mean, stdev
from tqdm import tqdm 
from joblib import Parallel, delayed
from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
 # import custom red agents
from FSRedAgentAggressiveTargetA import FSRedAgentAggressiveTargetA
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
import torch

from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from datetime import datetime
import json
import sys
import os

from wrapper.observation_graph import ObservationGraph
from ray.rllib.env.multi_agent_env import MultiAgentEnv
''' Check where the saved folder is to avoid overwriting
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# مدل لود شده رو چک کن ###############################
'''
from models.cage4 import InductiveGraphPPOAgent
from models.cage4 import load
from wrapper.graph_wrapper import GraphWrapper

# from models.cage4_onehot_attention import InductiveGraphPPOAgent
# from models.cage4_onehot_attention import load
# from wrapper.graph_wrapper_onehot_attention import GraphWrapper

''' SWITCH IMPORTS AS WELL'''


class Submission:

    # Submission name
    NAME: str = "Model Specific"
    TEAM: str = "Cybermonic-ModelSpec"
    TECHNIQUE: str = "PPO + Model Specific"
    '''domain specific''' #shouldn't be Oct10?
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Oct3_{i}_checkpoint.pt')
    #     for i in range(5)
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Oct4_{i}_checkpoint.pt')
    #     for i in range(5)
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Oct9_{i}_checkpoint.pt')
    #     for i in range(5) #Sleep Agent reward -33!!
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Oct10_{i}_checkpoint.pt')
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Nov26_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedCombined_Nov30_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedTargetA_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/gnn_ppo-{i}.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedCombined_Dec5_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedTargetBOperational_Dec6.py_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedTargetA_Dec5_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedDegrader_Dec6_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/defense_baselines/Dave_IPPO/actor_ppo_{i}') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedDegrader_Dec14_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
    #     for i in range(5) 
    # }
    AGENTS = {
        f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/results_DR_cybermonic/DR_cybermonic_Dec15_{i}_checkpoint.pt') #The model trained on FS, Impacter, degrader, failure tracker, combined Red agents.
        for i in range(5) 
    }
    # print(f'agent paths: {os.path.dirname(__file__)}/checkpoints_modelspec/gnn_ppo-i.pt')
    
    @staticmethod
    def wrap(env: CybORG, red_agent_class) -> MultiAgentEnv:
            print(f'red id check: {RED_AGENT_DICT.get(red_agent_class)}')
            ''' change wrapper''' 
            # return GraphWrapper(env, attacker_id=RED_AGENT_DICT.get(red_agent_class), total_attaacker_num=12)
            return GraphWrapper(env)



cyborg_version = CYBORG_VERSION
EPISODE_LENGTH = 500


# List of all red agent classes you want to evaluate
RED_AGENT_CLASSES = [
    # FSRedAgentCombined,
    FSRedAgentDegrader,
    FSRedAgentFailureTracker,
    FSRedAgentImpacter,
    FSRedAgentTargetA,
    FSRedAgentTargetAOperational,
    FSRedAgentTargetARestricted,
    FSRedAgentTargetB,
    FSRedAgentTargetBOperational,
    FSRedAgentTargetBRestricted,
    FiniteStateRedAgent,
    SleepAgent,
]
RED_AGENT_DICT = {
    SleepAgent: 0,
    FSRedAgentCombined: 1,
    FSRedAgentDegrader: 2,
    FSRedAgentFailureTracker: 3,
    FSRedAgentImpacter: 4,
    FSRedAgentTargetA: 5,
    FSRedAgentTargetAOperational: 6,
    FSRedAgentTargetARestricted: 7,
    FSRedAgentTargetB: 8,
    FSRedAgentTargetBOperational: 9,
    FSRedAgentTargetBRestricted: 10,
    FiniteStateRedAgent: 11
}

def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)


def load_submission(source: str):
    return Submission()


def evaluate_one_episode(cyborg, wrapped_cyborg, agent, write_to_file, i,tot):
    observations, _ = wrapped_cyborg.reset()
    r = []
    a = []
    o = []
    count = 0
    for j in tqdm(range(EPISODE_LENGTH), desc=f'({i+1}/{tot})'):
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], wrapped_cyborg.action_space(agent_name)
            )
            for agent_name, agent in submission.AGENTS.items()
            if agent_name in wrapped_cyborg.agents
        }
        observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
        done = {
            agent: term.get(agent, False) or trunc.get(agent, False)
            for agent in wrapped_cyborg.agents
        }
        if all(done.values()):
            break
        r.append(mean(rew.values()))

        if write_to_file:
            a.append(
                {
                    agent_name: cyborg.get_last_action(agent_name)
                    for agent_name in wrapped_cyborg.agents
                }       
            )
            o.append(
                {
                    agent_name: observations[agent_name]
                    for agent_name in observations.keys()
                }
            )
    total_reward = sum(r)
    return total_reward, a, o, r

def run_evaluation_parallel(submission, red_agent_class, log_path, max_eps=100, write_to_file=False, seed=None, workers=32):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    envs = []
    for _ in range(workers):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=red_agent_class,

            # red_agent_class=SleepAgent,
            # red_agent_class=FSRedAgentCombined,
            # red_agent_class=FSRedAgentDegrader,
            # red_agent_class=FSRedAgentFailureTracker,
            # red_agent_class=FSRedAgentImpacter,
            # red_agent_class=FSRedAgentTargetA,
            # red_agent_class=FSRedAgentTargetAOperational,
            # red_agent_class=FSRedAgentTargetARestricted,
            # red_agent_class=FSRedAgentTargetB,
            # red_agent_class=FSRedAgentTargetBOperational,
            # red_agent_class=FSRedAgentTargetBRestricted,
            # red_agent_class=FiniteStateRedAgent,
            steps=EPISODE_LENGTH,
        )
        cyborg = CybORG(sg, "sim", seed=seed)
        wrapped_cyborg = submission.wrap(cyborg, red_agent_class)
        envs.append((cyborg, wrapped_cyborg))
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    outs = Parallel(prefer='processes', n_jobs=workers)(
        delayed(evaluate_one_episode)(*envs[i % workers], submission.AGENTS, write_to_file, i, max_eps)
        for i in range(max_eps)
    )
    total_reward, actions_log, obs_log, step_reward_log = zip(*outs)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")
            
        ''' write step-wise reward to file'''    
        with open(log_path + "step_rewards.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            for ep_idx, ep_rewards in enumerate(step_reward_log):
                data.write(f"Episode {ep_idx} step_rewards: {ep_rewards}\n")

def run_evaluation(submission, red_agent_class, log_path, max_eps=100, write_to_file=False, seed=None):
    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = 500
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class = red_agent_class,
        # red_agent_class=SleepAgent,
        # red_agent_class=FSRedAgentCombined,
        # red_agent_class=FSRedAgentDegrader,
        # red_agent_class=FSRedAgentFailureTracker,
        # red_agent_class=FSRedAgentImpacter,
        # red_agent_class=FSRedAgentTargetA,
        # red_agent_class=FSRedAgentTargetAOperational,
        # red_agent_class=FSRedAgentTargetARestricted,
        # red_agent_class=FSRedAgentTargetB,
        # red_agent_class=FSRedAgentTargetBOperational,
        # red_agent_class=FSRedAgentTargetBRestricted,
        # red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg, red_agent_class)
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")

    start = datetime.now()

    total_reward = []
    actions_log = []
    obs_log = []
    step_reward_log = []

    
    for i in tqdm(range(max_eps)):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        count = 0
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            if write_to_file:
                a.append(
                    {
                        agent_name: cyborg.get_last_action(agent_name)
                        for agent_name in wrapped_cyborg.agents
                    }       
                )
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
        total_reward.append(sum(r))

        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)
            step_reward_log.append(r)

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward)
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                },
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
            }
            json.dump(data, output)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")

        ''' write step-reward file'''
        with open(log_path + "step_rewards.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            for ep_idx, ep_rewards in enumerate(step_reward_log):
                data.write(f"Episode {ep_idx} step_rewards: {ep_rewards}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Evaluation Script")
    parser.add_argument(
        "--append-timestamp",
        action="store_true",
        help="Appends timestamp to output_path",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set the seed for CybORG"
    )

    # Added to speed up evaluation 
    parser.add_argument(
        '--distribute', type=int, default=1, help="How many parallel workers to use"
    )
    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")
    args = parser.parse_args()
    
    '''specify the save folder
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
'''
    # args.output_path = os.path.abspath('results')
    # args.output_path = os.path.abspath('logs_onehot_attention')
    # args.output_path = os.path.abspath('checkpoints_modelspec')
    # args.output_path = os.path.abspath('checkpoints_cybermonic')
    args.output_path = os.path.abspath('results_DR_cybermonic')
    print(f'output path { args.output_path }')

    args.submission_path = os.path.abspath('')

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    if args.append_timestamp:
        args.output_path += time.strftime("%Y%m%d_%H%M%S") + "/"

    rmkdir(args.output_path)

    submission = load_submission(args.submission_path)


    for RAC in RED_AGENT_CLASSES:
        red_agent_class = RAC.__name__
        print("\n==============================")
        print(f" Evaluating Red Agent: {red_agent_class}")
        print("==============================\n")
    
        # Make a directory for this agent
        agent_output_path = os.path.join(args.output_path, red_agent_class)
        rmkdir(agent_output_path)
        
        if args.distribute == 1:
            run_evaluation(
                submission,RAC, max_eps=args.max_eps, log_path=agent_output_path, seed=args.seed, write_to_file=True
            )
        else: 
            run_evaluation_parallel(
                submission,RAC, max_eps=args.max_eps, log_path=agent_output_path, seed=args.seed, workers=args.distribute, write_to_file= True
            )



