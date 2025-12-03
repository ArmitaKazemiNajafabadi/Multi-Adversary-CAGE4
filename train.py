import sys
print(sys.executable)  # Should show your venv path
from models.cage4 import load
import random
from argparse import ArgumentParser
import os 
from types import SimpleNamespace
import math
from joblib import Parallel, delayed
import torch
from tqdm import tqdm

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

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

from models.cage4 import InductiveGraphPPOAgent
from models.memory_buffer import MultiPPOMemory
from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph

SEED = 1337
HYPER_PARAMS = SimpleNamespace(
    N = 25,             # How many episodes before training
    workers = 25,       # How many envs can run in parallel
    bs = 2500,          # How many steps to learn from at a time
    episode_len = 500,
    training_episodes = 500_000, # Realistically, stops improving around 50k
    epochs = 4
)

N_AGENTS = 5 
MAX_THREADS = 36 # 5 per subnet (20 for agent 4, 4 for all others)
torch.manual_seed(SEED)
torch.set_num_threads(MAX_THREADS)

@torch.no_grad()
def generate_episode_job(agents, env, hp, i): 
    '''
    Per-process job to generate one episode of memories
    for all 5 agents. Returns `N_AGENTS` memory buffers, 
    and the total reward for the episode. 

    Args: 
        agents:     list of keep.cage4.InductiveGraphAgent objects 
        env:        wrapped cyborg object 
        hp:         hyperparameter namespace 
        i:          process id in range(0, `hp.workers`)
    '''
    torch.set_num_threads(MAX_THREADS // hp.workers)

    # Initialize environment
    env.reset()
    states = env.last_obs
    # for key in states:
    #     print(states.get(ke))
    #     break
    blocked_rewards = [0]*N_AGENTS

    tot_reward = 0
    memory_buffers = MultiPPOMemory(hp.bs)

    # Begin episode 
    for ts in tqdm(range(hp.episode_len), desc=f'Worker {i}'):
        actions = dict()
        memories = dict()

        # Get actions for all unblocked agents
        for k,(state,blocked) in states.items():
            i = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action,value,prob = agents[i].get_action((state,blocked))
                memories[i] = (state,action,value,prob)
                actions[k] = action

        next_state, rewards, _,_,_ = env.step(actions)
        rewards = list(rewards.values())
        tot_reward += sum(rewards)/N_AGENTS

        # Delay recieving rewards until multi-step actions are completed. 
        # Agents recieve cumulative reward for all the timesteps 
        # they spent performing their action. 
        for i in range(N_AGENTS):
            if i in memories:
                s,a,v,p = memories[i]
                r = rewards[i] + blocked_rewards[i]
                t = 0 if ts < hp.episode_len-1 else 1

                memory_buffers.remember(i, s,a,v,p, r,t)
                blocked_rewards[i] = 0
            else:
                blocked_rewards[i] += rewards[i]
        
        states = next_state

    return memory_buffers.mems, tot_reward

def train(agents, hp, seed=SEED): #Editted to enable random RedAgent choice 
    [agent.train() for agent in agents]
    log = []

    # Red agent options
    # red_agents = [SleepAgent, FSRedAgentCombined, FSRedAgentDegrader,
    #     FSRedAgentFailureTracker, FSRedAgentImpacter,
    #     FSRedAgentTargetA, FSRedAgentTargetAOperational, FSRedAgentTargetARestricted,
    #     FSRedAgentTargetB, FSRedAgentTargetBOperational, FSRedAgentTargetBRestricted,
    #     FiniteStateRedAgent          
    # ]
    red_agents = [FSRedAgentTargetBOperational]
    # red_agents = [FSRedAgentAggressiveTargetA, FSRedAgentCombined, FSRedAgentDegrader,
    #     FSRedAgentFailureTracker, FSRedAgentImpacter,
    #     FSRedAgentTargetA, FSRedAgentTargetAOperational, FSRedAgentTargetARestricted,
    #     FSRedAgentTargetB, FSRedAgentTargetBOperational, FSRedAgentTargetBRestricted,
    #     FiniteStateRedAgent          
    # ]
    def create_envs():
        """Helper function to create environments with random red agents"""
        envs = []
        for i in range(min(hp.workers, hp.N)):
            red_agent = random.choice(red_agents)
            sg = EnterpriseScenarioGenerator(
                blue_agent_class=SleepAgent,
                green_agent_class=EnterpriseGreenAgent,
                red_agent_class=red_agent,
                steps=hp.episode_len,
            )
            env = CybORG(sg, "sim", seed=seed)
            envs.append(GraphWrapper(env))
        return envs
    # Initial environment creation
    envs = create_envs()
    
    # Training loop
    
    # Define learn function for threads to call later so we can 
    # parallelize the backprop step. Use more threads for Agent 4 
    # because they're managing 3 subnets instead of 1 (bigger graph/matrices)
    # Still not perfectly load-balanced, but close enough
    def learn(i):
            if i < 4:
                torch.set_num_threads(MAX_THREADS // 9)
            else:
                torch.set_num_threads((MAX_THREADS // 9) * N_AGENTS)
            return agents[i].learn()

    # Begin training loop 
    best_avg = -math.inf
    best_avg_log = []
    for e in tqdm(range(hp.training_episodes // hp.N)):
        e *= hp.N

        # Recreate environments every 100 iterations (2,500 episodes)
        if e > 0 and (e // hp.N) % 15 == 0:
            print(f"Recreating environments with new red agents at episode {e}")
            envs = create_envs()
            
        # Generate N episodes in parallel 
        out = Parallel(prefer='processes', n_jobs=hp.workers)(
            delayed(generate_episode_job)(agents, envs[i % len(envs)], hp, i) for i in range(hp.N)
        )

        # Concat memories across episodes, and transfer them to agents' 
        # internal memory buffers 
        memories, avg_rewards = zip(*out)
        memories = [list(m) for m in zip(*memories)]
        for i in range(N_AGENTS):
            agents[i].memory.mems = memories[i]

        # Use threads because agents are in heap memory 
        # Parallel backpropagation 
        print("Updating step ..." + str(e))
        last_losses = Parallel(prefer='threads', n_jobs=N_AGENTS)(
            delayed(learn)(i) for i in range(N_AGENTS)
        )

        losses = ','.join([f'{last_losses[i]:0.4f}' for i in range(N_AGENTS)])
        # print(f"[{e}] Loss: [{losses}]")

        # Log average reward across all episodes 
        avg_reward = sum(avg_rewards) / hp.N
        print(f"Avg reward for episode: {avg_reward}")
        log.append((avg_reward,e,sum(last_losses)/N_AGENTS))
        torch.save(log, f'checkpoints_modelspec/{hp.fnames}.pt')

               # Checkpoint model states 
        if avg_reward>best_avg:
            for i in range(N_AGENTS):
                agent = agents[i]
                agent.save(outf=f'checkpoints_modelspec/{hp.fnames}_{i}_checkpoint.pt')
            best_avg = avg_reward
            print(f"Avg reward for episode: {avg_reward}")
            best_avg_log.append((avg_reward,e,sum(last_losses)/N_AGENTS))
            torch.save(log, f'checkpoints_modelspec/best_avg_{hp.fnames}.pt')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('fname', help='Required: the name to save output files as.')
    ap.add_argument('--hidden', action='store', type=int, default=256, help='Dimension of middle layer for actor/critic')
    ap.add_argument('--embedding', action='store', type=int, default=128, help='Dimension of node representation for actor/critic')

    args = ap.parse_args()
    print(args)

    # Add directory for files
    if not os.path.exists('checkpoints_modelspec'):
        os.mkdir('checkpoints_modelspec')


    # Add 5 extra dimensions to observation graph: 
    #   2 for tabular data (gets appended to relevant hosts)
    #   3 for message data (gets appended to relevant subnets): 
    #       1 bit if subnet has comprimised host in it
    #       1 bit if subnet has scanned host in it
    #       1 bit if message was sent successfully 
    
    # All handled in wrapper.graph_wrapper
    agents = [InductiveGraphPPOAgent(
        ObservationGraph.DIM+5,
        bs=HYPER_PARAMS.bs,
        a_kwargs={'lr': 0.0003, 'hidden1': args.hidden, 'hidden2': args.embedding},
        c_kwargs={'lr': 0.001, 'hidden1': args.hidden, 'hidden2': args.embedding},
        clip=0.2,
        epochs=HYPER_PARAMS.epochs
    ) for _ in range(N_AGENTS)]
    # for i in range(N_AGENTS):
    #         data = torch.load(f'{os.path.dirname(__file__)}/checkpoints_modelspec/fsRedFailureTrackerNov30-{i}_checkpoint.pt')
    #         agents[i].actor.load_state_dict(data['actor'])
    #         agents[i].critic.load_state_dict(data['critic'])
    # agents = [load(f'{os.path.dirname(__file__)}/checkpoints/Sep10Train-{i}_checkpoint.pt') for i in range(N_AGENTS)]
    HYPER_PARAMS.fnames = args.fname
    train(agents, HYPER_PARAMS)