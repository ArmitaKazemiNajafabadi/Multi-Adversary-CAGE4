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
from SleepSwitchFSRandom import SleepSwitchFSRandom
from ImpacterSwitchFSRandom import ImpacterSwitchFSRandom
from FSMultipleSwitchAgent1 import FSMultipleSwitchAgent1
from FSMultipleSwitchAgent2 import FSMultipleSwitchAgent2
'''only the parallel evaluation logs the belief'''


''' Check where the saved folder is to avoid overwriting
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
############################################# SPECIFY THE SAVE FOLDER CORRECTLY ###############################
'''


from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from datetime import datetime

import json

import sys
import os

cyborg_version = CYBORG_VERSION
EPISODE_LENGTH = 500


# List of all red agent classes you want to evaluate
RED_AGENT_CLASSES = [
    # FSRedAgentAggressiveTargetA,
    # FSRedAgentCombined,
    # FSRedAgentDegrader,
    # FSRedAgentFailureTracker,
    # FSRedAgentImpacter,
    # FSRedAgentTargetA,
    # FiniteStateRedAgent,
    # SleepAgent,
    # FSRedAgentTargetAOperational,
    # FSRedAgentTargetARestricted,
    # FSRedAgentTargetB,
    # FSRedAgentTargetBOperational,
    # FSRedAgentTargetBRestricted,
    # SleepSwitchFSRandom,
    # ImpacterSwitchFSRandom,
    # FSMultipleSwitchAgent1,
    FSMultipleSwitchAgent2
]

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
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission.submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission


def evaluate_one_episode(cyborg, wrapped_cyborg, agent, write_to_file, i,tot):
    observations, _ = wrapped_cyborg.reset()
    r = []
    a = []
    o = []
    belief_log = []
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

        ''' add belief capture'''
        belief_log.append({
        "t": j,
        "id": red_agent_class,
        "belief": wrapped_cyborg.get_belief()
        })
        ''' end belied capture'''
        
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
    return total_reward, a, o, r, belief_log

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
        wrapped_cyborg = submission.wrap(cyborg)
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
    total_reward, actions_log, obs_log, step_reward_log, belief_log = zip(*outs)

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


        ''' write belief dist. to file'''    
        with open(log_path + "belief_log.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            for ep_idx, ep_beliefs in enumerate(belief_log):
                data.write(f"Episode {ep_idx} beliefs: {ep_beliefs}\n")


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
    wrapped_cyborg = submission.wrap(cyborg)
    
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
    belief_log = []

    
    for i in tqdm(range(max_eps)):
        observations, _ = wrapped_cyborg.reset()
        r = []
        a = []
        o = []
        count = 0
        # inside the episode loop (before the step loop):
        ep_belief_log = []  #'''added'''
        for j in range(EPISODE_LENGTH):
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)


            ''' add belief capture'''
            ep_belief_log.append({
                "t": j,
                "id": red_agent_class,
                "belief": wrapped_cyborg.get_belief()
            })
            ''' end belied capture'''

        
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
            belief_log.append(ep_belief_log)  # see note below


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

        ''' write belief dist. to file'''    
        with open(log_path + "belief_log.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            for ep_idx, ep_beliefs in enumerate(belief_log):
                data.write(f"Episode {ep_idx} beliefs: {ep_beliefs}\n")


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
    args.output_path = os.path.abspath('results')
    # args.output_path = os.path.abspath('logs_onehot_attention')
    
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