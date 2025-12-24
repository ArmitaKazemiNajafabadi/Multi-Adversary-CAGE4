''' file created on Dec 24, 2025, for quick_eval.py of a couple of proposed model checkpoints eval"'''
from copy import deepcopy
from train_with_transformer_2 import TinyTransformerClassifier # Nov 21, added
from train_with_transformer_2 import random_crop_batch
from train_with_transformer_2 import train_classifier

import numpy as np
import torch
import hashlib
import torch.nn.functional as F

import os
from CybORG.env import CybORG
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from CybORG.Simulator.Actions.Action import Sleep
from CybORG.Shared.Enums import TernaryEnum

from wrapper.observation_graph import ObservationGraph
from wrapper.globals import *

class GraphWrapper(EnterpriseMAE):
    def __init__(self, env: CybORG, window_size, total_attaacker_num, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.graphs = dict()
        self.env = env
        self.agent_names = [f'blue_agent_{i}' for i in range(5)]
        self.ts = 0
        self.belief = torch.full((1, total_attaacker_num), 1/total_attaacker_num)  # 12 attacker types
        # print(self.belief.size())
        self.gen_obs = None
        self.L = window_size
        # print(f"initialize graph wrapper with belief: {self.belief}")
        self.stacked_obs = []
        '''Nov 21, 2025 switch'''
        self.classifier = TinyTransformerClassifier(input_dim=(129*4 + 247))
        self.classifier.load_state_dict(torch.load(f'/projects/ImaniLab/Armita/CAGE-Multi-Adversary/results/classifier.pt'))
        self.classifier.eval()

        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }
   
    ''' update_belief 21 Nov 2025 Version '''
    def update_belief(self, b_lr=0.05):
        MAX_SEQ = 200 if self.ts > 200 else self.ts                              #Change to 200 and re-test if needed
        seq_used = self.stacked_obs[-MAX_SEQ:]
        suitable_input = torch.stack(seq_used, dim=0)
        # --- APPLY NORMALIZATION — VERY IMPORTANT ---
    
        stats = torch.load("results/normalization_stats.pt")
        mean = stats["mean"]
        std = stats["std"]
        suitable_input = (suitable_input - mean) / std
        input_tensor = suitable_input.unsqueeze(0)      # [1, L, 763]
        with torch.no_grad():
            model_out = self.classifier(input_tensor) 
            probs = torch.softmax(model_out,dim=1)
        if self.ts <= 150:
            new_belief = (1.0 - b_lr) * self.belief + b_lr * probs
        else: 
            new_belief = probs.clone()
        new_belief = new_belief / new_belief.sum(dim=1, keepdim=True)
        self.belief = new_belief.clone() 
        # print(f"belief: {self.belief}")
        

    
    def action_translator(self, agent_name, a_id):
        session = 0 # Seems the same every time?
        if a_id is None:
            return Monitor(session, agent_name)
        agent_id = int(agent_name[-1])
        which_subnet = MY_SUBNETS[agent_id][a_id // MAX_ACTIONS]
        a_id %= MAX_ACTIONS
        # Node action
        if a_id < N_NODE_ACTIONS*MAX_HOSTS:
            a = NODE_ACTIONS[a_id // MAX_HOSTS]
            target = a_id % MAX_HOSTS
            if target > 5:
                target = f'{which_subnet}_user_host_{target-6}'
            else:
                target = f'{which_subnet}_server_host_{target}'
            return a(session=session, agent=agent_name, hostname=target)
        # Edge action
        elif (a_id := a_id - (N_NODE_ACTIONS*MAX_HOSTS)) < (N_EDGE_ACTIONS*POSSIBLE_NEIGHBORS):
            a = EDGE_ACTIONS[a_id // POSSIBLE_NEIGHBORS]
            target = [r for r in ROUTERS if which_subnet not in r][a_id % POSSIBLE_NEIGHBORS]
            return a(session, agent_name, target.replace('_router',''), which_subnet)
        # Global action (only one)
        else:
            return Monitor(session, agent_name)


    def step(self, action):
        # Convert from model out to Action objects
        action = {
            k:self.action_translator(k,v)
            for k,v in action.items()
        }
        """ does the order of running matter? It is right now. it wasn't right order before."""
        states_for_classifier = []
        for i in range(5):
            agent = f'blue_agent_{i}'
            # Get the raw observation dictionary for this agent
            dict_obs1 = self.env.environment_controller.get_last_observation(agent).data
            state_classifier = self.encode_dict_obs(dict_obs1, self.gen_obs[agent]).to('cpu')  # Keep on CPU when collecting
            states_for_classifier.append(state_classifier)
        ''' end '''
        # Gets the info from the tabular wrapper (4 dims per host, in order)
        observation, reward, term, trunc, info = super().step(
            action_dict=action, messages=self.msg
        )
        self.gen_obs = deepcopy(observation)

        # Tell ObservationGraph what happened and update
        graph_obs = dict()

        '''begin modifies'''
        # states_for_classifier = []
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = observation[agent]
            g = self.graphs[agent]

            # Get the raw observation dictionary for this agent
            dict_obs = self.env.environment_controller.get_last_observation(agent).data
            # state_classifier = self.encode_dict_obs(dict_obs, self.gen_obs[agent]).to('cpu')  # Keep on CPU when collecting
            # states_for_classifier.append(state_classifier)
            # '''not end modifies but in this loop it is an end'''
        
            msg = dict_obs.pop('message')
            msg = np.stack(msg, axis=0)

            # Indicates if msg was recieved or comms are blocked
            # This way we differentiate between feature for 0 and unknown
            recieved_msg = msg[:, -1:]
            if i != 4:
                # Repeat agent 4's 'is_recieved' message across 2 more subnets
                recieved_msg = np.concatenate([recieved_msg, np.zeros((2,1))], axis=0)
                recieved_msg[-2:] = recieved_msg[-3]

                # Pull out messages for 'was_scanned' and 'was_comprimised'
                msg_small = msg[:-1, :2]
                msg_big = msg[-1, :6].reshape(3,2)
                msg = np.concatenate([msg_small, msg_big], axis=0)
            else:
                msg = msg[:, :2]

            msg = np.concatenate([msg, recieved_msg], axis=1)

            # Update the graph based on the raw dictionary 
            g.parse_observation(dict_obs)

            # Pull node features from tabular observation, and also update 
            # graph subnet connectivity edges. 
            tab_x,phase,new_msg = self._parse_tabular(o, g) 
            
            self.msg[agent] = new_msg

            # Combine node features from graph source, and tabular source
            x,ei,masks = g.get_state(MY_SUBNETS[i])
            x = self._combine_data(x, tab_x)

            # Mask/pack into conviniently sized tensors for the GNN models 
            obs = self._to_obs(x,ei,masks,phase,msg,new_msg)

            # During training, we need to know if the agent is still mid-action
            # If so, we don't bother calculating an action next turn 
            is_blocked = dict_obs['success'] == TernaryEnum.IN_PROGRESS
            graph_obs[agent] = (obs, is_blocked)

        ''' 21 Nov 2025 Version '''
        concatenation = torch.cat(states_for_classifier, dim=1)
        self.stacked_obs.append(concatenation.squeeze(0))
        # self.update_belief()        
        self.update_belief()
        self.ts += 1
        self.last_obs = graph_obs
        return graph_obs, reward, term, trunc, info

    def reset(self):
        '''
        Rebuild internal graph representation with parameters of new environment
        '''
        self.ts = 0
        self.belief = torch.full((1, 12), 1/12)  # 12 attacker types
        self.stacked_obs = []  # ← ADD THIS
        obs_tab, action_mask = super().reset()
        '''added Dec 15'''
        self.gen_obs = deepcopy(obs_tab) 
        '''ended Dec 15'''

        g = ObservationGraph()

        obs_dict = self.env.environment_controller.init_state
        g.setup(obs_dict)

        # Set message to empty for all agents
        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        my_state = dict()
        self.graphs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = obs_tab[agent]

            # Message from agent 4 has 2 extra subnet infos
            if i != 4:
                dummy_msg = (np.zeros((6,3)), np.zeros(8))
            else:
                dummy_msg = (np.zeros((4,3)), np.zeros(8))

            # Duplicate shared observation of the initial graph across
            # all agents (but make sure not to pass by reference)
            g_ = deepcopy(g)
            self.graphs[agent] = g_

            # Get tabular features and update connectivity graph 
            tab_x,phase,_ = self._parse_tabular(o,g_)

            # Combine all node features together and package for agents
            x,ei,masks = g_.get_state(MY_SUBNETS[int(agent[-1])])
            x = self._combine_data(x, tab_x)
            obs = self._to_obs(x,ei,masks,phase, *dummy_msg)

            # By default, agents are not blocked on turn 0
            my_state[agent] = (obs, False)

        self.last_obs = my_state
        return my_state, action_mask


    def _parse_tabular(self, x, g):
        
        phase_idx = int(x[0])

    
        sn_block = x[1:-(4*8)]
        subnets = sn_block.shape[0] // SN_BLOCK_SIZE

        relevant_subnets = []
        src = []
        dst = []
        x = torch.zeros(g.n_permenant_nodes, 2)
        msgs = []

        # Only affects agent4 but may as well be generalizeable
        for i in range(subnets):
            block = sn_block[SN_BLOCK_SIZE*i : SN_BLOCK_SIZE*(i+1)]

            # Pull out edges between subnets
            sn = block[:18]
            me = ROUTERS[ sn[:9].nonzero()[0][0] ]
            can_maybe_connect_to = (sn[9:18] == 0).nonzero()[0]

            # Logic for subnet routing 
            if INTERNET in can_maybe_connect_to:
                can_connect_to = [ROUTERS[i] for i in can_maybe_connect_to]
            else:
                # Can connect to anything in LAN
                can_connect_to = [
                    ROUTERS[i] for i in can_maybe_connect_to
                    if ROUTERS[i] in ACCESSABLE_OFFLINE[me]
                ]

            router_name = me
            me = [me] * len(can_connect_to)
            src += can_connect_to
            dst += me

            # Pull out features for servers/hosts that exist
            hosts = torch.from_numpy(block[27:]).reshape(2,16).T
            n_srv, n_usr = g.subnet_size[router_name]
            srv_idx = list(range(n_srv))
            usr_idx = list(range(6,n_usr+6))

            # Insert into rows corresponding w server/host nodes in graph
            # (Always directly after node for subnet they are on)
            start_usr_idx = g.nids[router_name]+1
            start_srv_idx = start_usr_idx + len(usr_idx)
            end_srv_idx = start_srv_idx + len(srv_idx)

            # Note: TabularWrapper goes from server to host, but
            # graph goes from host to server (alphabetically)
            # so we have to do some lifting to rearrange
            x[start_usr_idx : start_srv_idx] = hosts[usr_idx]
            x[start_srv_idx : end_srv_idx] = hosts[srv_idx]

            # Each subnet can add 2 bits to the message for if any hosts
            # are compromised/have been scanned
            msg = list((hosts.sum(dim=0) > 0).long())
            msgs += msg

            relevant_subnets.append(router_name)

        g.set_firewall_rules(src,dst)
        phase = torch.zeros((1,3))
        phase[0,phase_idx] = 1
        """ modified part """
        phase = torch.cat([phase, self.belief], dim=1)  # Shape: [1, 15]
        """ modified part """

        # Make messages all 8-dim and add checkbit to the end
        padding = 8-len(msgs)
        msgs += [0]*padding
        msgs[-1] = 1
        msg = np.array(msgs)

        return x,phase,msg

    def _combine_data(self, graph_x, tabular_x):
        '''
        Stick the tabular data onto the node feature matrix 
        on the appropriate rows--those corresponding with 
        the hosts the tabular data is referencing 
        '''
        # Tabular x only accounts for subnets and workstations
        # Processes/connections have higher indices, but no features
        # from the FlatActionWrapper, so need to be padded before combined
        padding = torch.zeros(
            graph_x.size(0) - tabular_x.size(0),
            tabular_x.size(1)
        )
        tabular_x = torch.cat([tabular_x, padding], dim=0)

        return torch.cat([graph_x, tabular_x], dim=1)

    def _to_obs(self, x,ei,masks,phase, other_msg,my_msg):
    
        if len(masks) == 1:
            (srv,usr,edge,rtrs) = masks[0]

            all_msg = torch.zeros((x.size(0), 3))
            all_msg[rtrs] = torch.from_numpy(other_msg).float()

            # Edge[0][0] is always the subnet node managed by this agent
            all_msg[edge[0][0], :2] = torch.from_numpy(my_msg[:2]).float()

            # Set 'is_recieved' to a special value to indicate this is self
            all_msg[edge[0][0], 2] = -1

            x = torch.cat([x, all_msg], dim=1)
            '''added part'''

            belief_expanded = self.belief.repeat(x.size(0), 1)  # [num_nodes, total_attacker_num]
            x = torch.cat([x,belief_expanded], dim=1)

            '''end added part'''
            return (
                x,ei,phase,
                srv,torch.tensor([srv.size(0)]),
                usr,torch.tensor([usr.size(0)]),
                edge, False
            )


        srv,usr,edges = [],[],[]
        n_srv,n_usr = [],[]
        my_ids = []
        for (s,u,e,_) in masks:
            my_ids.append(e[0][0].item())

            srv.append(s)
            usr.append(u)
            edges.append(e)

            n_srv.append(s.size(0))
            n_usr.append(u.size(0))


    
        rtrs = masks[0][3]
        other_rtrs = [o.item() for o in rtrs if o.item() not in my_ids]
        all_msg = torch.zeros(x.size(0), 3)

        all_msg[other_rtrs] = torch.from_numpy(other_msg).float()
        all_msg[my_ids, :2] = torch.from_numpy(my_msg[:6].reshape(3,2)).float()
        all_msg[my_ids, 2] = -1

        x = torch.cat([x, all_msg], dim=1)
        
        '''added part'''
        belief_expanded = self.belief.repeat(x.size(0), 1)  # [num_nodes, total_attacker_num]

        x = torch.cat([x,belief_expanded], dim=1)
        '''end added part'''
        return (
            x,ei,phase.repeat_interleave(3,0),
            torch.cat(srv), torch.tensor(n_srv),
            torch.cat(usr), torch.tensor(n_usr),
            torch.cat(edges, dim=1), True
        )

    def encode_dict_obs(self, dict_obs, gen_obs):
        # print(f'type(gen_obs){type(gen_obs)} + shape: {gen_obs.shape}')
        # Map based on enum name strings
        success_map = {
            "TRUE":        [1, 0, 0, 0],
            "FALSE":       [0, 1, 0, 0],
            "UNKNOWN":     [0, 0, 1, 0],
            "IN_PROGRESS": [0, 0, 0, 1],
        }
        action = dict_obs.get('action', "Monitor")
        action_hashed = string_to_range(action)
        # print(f"action hasehd {action_hashed}")

        # Use enum.name for lookup
        success_enum = dict_obs.get('success', TernaryEnum.UNKNOWN)
        success_encoding = success_map.get(success_enum.name, [0, 0, 1, 0])  # fallback = UNKNOWN
    
        # Flatten the 4 message arrays
        message_vec = []
        for arr in dict_obs.get('message', []):
            message_vec.extend(arr.astype(np.float32).tolist())
    
        if len(message_vec) < 32:
            message_vec += [0.0] * (32 - len(message_vec))
    
        # Combine all features    
        features = [action_hashed] + success_encoding + message_vec  + gen_obs.tolist() 
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        # print("Feature tensor shape:", features_tensor.shape)
        
        return features_tensor
    
def string_to_range(s):
    """
    Maps a string deterministically to an integer in [0, N-1]
    """
    h = hashlib.sha256(str(s).encode('utf-8')).hexdigest()
    return (int(h, 16) % 250)/100
    