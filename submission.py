import os
import torch
# from models.cage4 import load
# from models.cage4_onehot import load
# from models.cage4_onehot_modified import load
# from models.cage4_onehot_att import load
from models.cage4_onehot_attention import load

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents import SleepAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from CybORG.Agents.Wrappers import EnterpriseMAE
# from wrapper.graph_wrapper import GraphWrapper 
# from wrapper.graph_wrapper_onehot import GraphWrapper
# from wrapper.graph_wrapper_onehot_attention import GraphWrapper
# from wrapper.wrapper_Meta import GraphWrapper
from wrapper.graph_wrapper_with_classifier_for_evaluation import GraphWrapper
### Import custom agents here ###
# from DummyAgent import DummyAgent
# from UnsupervisedAgent import UnsupervisedAgent
# from wrapper.observation_graph import ObservationGraph
from models.cage4_onehot_attention import InductiveGraphPPOAgent
# from models.cage4 import InductiveGraphPPOAgent


class Submission:

    # Submission name
    # NAME: str = "dummy"
    # NAME: str = "sleep"
    # NAME: str = "unsupervised"
    # NAME: str = "TEST TrainedBlueAgent"
    NAME: str = "KEEP"
    # NAME: str = "with Classifier"

    # Name of your team
    # TEAM: str = "Sleep"
    # TEAM: str = "Dummy"
    # TEAM: str = "Unsupervised-GPT"
    TEAM: str = "Cybermonic"
    # TEAM: str = "Cybermonic-Modified"
    # TEAM: str = "Ours"




    # What is the name of the technique used? (e.g. Masked PPO)
    # TECHNIQUE: str = "No Action"
    # TECHNIQUE: str = "Last Action"
    # TECHNIQUE: str = "Clustering"
    # TECHNIQUE: str = "PPO GPT"
    # TECHNIQUE: str = "Graph-based PPO With Intra-agent Communication"
    # TECHNIQUE: str = "Domain Randomization"
    TECHNIQUE: str = "PPO + Classifier"


    # Use this function to define your agents.
    # AGENTS: dict[str, BaseAgent] = {
    #     f"blue_agent_{agent}": UnsupervisedAgent(f"Agent{agent}") for agent in range(5)
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/policy_state_{i}.pkl')
    #     for i in range(5)
    # }
    # AGENTS: dict[str, BaseAgent] = {
    #     f"blue_agent_{agent}": TrainedBlueAgent(f"Agent{agent}", np_random=10) for agent in range(5)
    # }

    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/gnn_ppo-{i}.pt')
    #     for i in range(5)
    # }
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints/Sep17Train-{i}_checkpoint.pt')
    #     for i in range(5)
    # }
    '''domain randomization?''' #Oct2 without SleepAgent? #should try Oct4, nashod Oct3
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints/Oct3-{i}_checkpoint.pt')
    #     for i in range(5)
    # }
    
    '''domain specific''' #shouldn't be Oct10?
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_attention/Oct3_{i}_checkpoint.pt')
    #     for i in range(5)
    # }

    
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/Oct14_{i}_checkpoint.pt')
    #     for i in range(5)
    # }
    
    '''Nov 21 switch'''
    # AGENTS = {
    #     f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/checkpoints_onehot_att/Oct18_{i}_checkpoint.pt')
    #     for i in range(5)
    # }

    ''' ours '''
    AGENTS = {
        f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/results/agent_{i}_checkpoint.pt')
        for i in range(5)
    }
    
    # agents = [InductiveGraphPPOAgent(
    #     ObservationGraph.DIM+5+12,
    #     bs=2500,
    #     a_kwargs={'lr': 0.0003, 'hidden1': 320, 'hidden2': 128},
    #     c_kwargs={'lr': 0.001, 'hidden1': 320, 'hidden2': 128},
    #     clip=0.2,
    #     epochs=4
    # ) for _ in range(5)]
    # for i in range(5):
    #         data = torch.load(f'{os.path.dirname(__file__)}/weights/Oct12_{i}_checkpoint.pt')
    #         agents[i].actor.load_state_dict(data['actor'])
    #         agents[i].critic.load_state_dict(data['critic'])
    #         agents[i].eval()
        
    # AGENTS = {
    #     f"blue_agent_{i}": agents[i]
    #     for i in range(5)
    # }


    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        # return EnterpriseMAE(env)
        # return GraphWrapper(env)
        # return GraphWrapper(env, attack_id=1, total_attaacker_num=12)
        # return GraphWrapper(env, attacker_id=1, total_attaacker_num=12)
        return GraphWrapper(env, window_size=10 ,total_attaacker_num=12)
        # return GraphWrapper(env,total_attaacker_num=12)
