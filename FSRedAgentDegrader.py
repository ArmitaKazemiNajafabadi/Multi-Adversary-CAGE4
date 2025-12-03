from inspect import signature
from typing import Union, List, Dict
from pprint import pprint
from ipaddress import IPv4Address
from numpy import invert
import numpy as np

from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Simulator.Actions.AbstractActions import DiscoverRemoteSystems, PrivilegeEscalate, Impact, DegradeServices, AggressiveServiceDiscovery, StealthServiceDiscovery, DiscoverDeception
from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import PIDSelectiveExploitActionSelector, ExploitRemoteService
from CybORG.Simulator.Actions.ConcreteActions.RedSessionCheck import RedSessionCheck
from CybORG.Simulator.Actions.ConcreteActions.Withdraw import Withdraw
from CybORG.Simulator.Actions import Sleep, Action, InvalidAction
import copy

"""         

"""
class FSRedAgentDegrader(FiniteStateRedAgent):
    
    def __init__(self, name=None, np_random=None, agent_subnets=None):
        
        super().__init__(name, np_random)
        self.step = 0
        self.action_params = None
        self.last_action = None
        self.host_states = {} #has the IPs as keys and 'hostnames' associated, and what 'state' each host is in.
        self.host_service_decoy_status = {} #ports associated to each IP
        self.agent_subnets = agent_subnets #was mostly static
        self.action_list = super().action_list()
        self.host_state_sequence = []
        self.print_action_output = False
        self.print_obs_output = False
        
        self.prioritise_servers = False

        self.host_states_priority_list = super().set_host_state_priority_list()
        self.state_transitions_success = super().state_transitions_success() #what to do when success
        self.state_transitions_failure = super().state_transitions_failure() #what to do when fail
        self.state_transitions_probability = super().state_transitions_probability()    

    def get_action(self, observation: dict, action_space):   
        action = None
        success = None

        if 'success' in observation.keys():
            success = observation.pop('success')

        if 'action' in observation.keys():
            action = observation.pop('action')

        super()._host_state_transition(action, success)
        super()._process_new_observations(observation)
        super()._session_removal_state_change(observation)

        if self.print_action_output:
            super().last_turn_summary(observation, action, success)

        if success.name == 'IN_PROGRESS':
            self.step += 1
            return Sleep()
        else:
            known_hosts = [h for h in self.host_states.keys() if not self.host_states[h]['state'] == 'F']
            self.host_state_sequence.append(copy.deepcopy(self.host_states))
            # if self.step == 50:
                # for jjj in range(len(self.host_state_sequence)):
                    # print(self.host_state_sequence[jjj])
                    # print()
            chosen_host, action = self._choose_host_and_action(action_space, known_hosts)

            if isinstance(action, ExploitRemoteService) and chosen_host in list(self.host_service_decoy_status.keys()):
                action.exploit_action_selector = PIDSelectiveExploitActionSelector(excluded_pids=self.host_service_decoy_status[chosen_host])

            self.step += 1
            self.last_action = action
            return action

                        
    def _choose_host(self, host_options: List[str]):
        """A valid host is selected and returned"""
        if self.host_states_priority_list is None:
            state_host_options = host_options

        else:
            print("Not implemented")

      
        available_hosts = {};
        for h_opt in host_options:
            # print(f"self.host_stats: {self.host_states}")
            state = self.host_states[h_opt]['state']
            weight = 1

            hostname = self.host_states[h_opt].get('hostname')
            if state in ('R', 'RD'):
                weight *= 10  # boost impactful
            if not hostname in available_hosts.keys():
                available_hosts[h_opt] = weight
            else:
                # keep max weight if multiple hosts with same state
                available_hosts[h_opt] = max(available_hosts[h_opt], weight)
            if len(available_hosts) == len(state_host_options):
                break

        base = sum(available_hosts.values())
        # print(base)
        probs = [(p/base) for p in available_hosts.values()]
        # if bool: 
            # print(f"probes: {probs}")
        chosen_host = self.np_random.choice(list(available_hosts.keys()), p=probs)
        # chosen_host = self.np_random.choice(state_host_options)
        # print(chosen_host)

        return chosen_host
    
    
    def _choose_host_and_action(self, action_space: dict, host_options: List[str]):
        """The selection of a valid host and action to execute this step."""
        chosen_host = self._choose_host(host_options)
        if chosen_host == None:
            return Sleep()

        host_action_options = {self.action_list[i]: prob for i, prob in enumerate(self.state_transitions_probability[self.host_states[chosen_host]['state']]) if not prob == None}
        # print(host_action_options)
        invalid_actions = []
        # if chosen host not work, goes to all other host_options
        while True:
            options = [i for i, v in action_space['action'].items() if v and i not in invalid_actions and i in list(host_action_options.keys())]
            if len(options) > 0:
                hostname = self.host_states[chosen_host].get('hostname')
                probabilities = []
                for opt in options:
                    name = opt.__name__.lower()
                    if "degrade" in name:
                        host_action_options[opt] *= 4


                    probabilities.append(host_action_options[opt])

                action_class = self.np_random.choice(options, p=np.array(probabilities)/sum(probabilities))
            else:
                print("not implemented 2") #no prints
 
            # select random options
            action_params = {}
            for param_name in self.action_params[action_class]:
                options = [i for i, v in action_space[param_name].items() if v]
                if param_name == 'hostname':
                    if not self.host_states[chosen_host]['hostname'] == None:
                        action_params[param_name] = self.host_states[chosen_host]['hostname']
                    else:
                        invalid_actions.append(action_class)
                        action_params = None
                        break
                elif param_name == 'ip address' or param_name == "ip_address":
                    action_params[param_name] = IPv4Address(chosen_host)
                elif len(options) > 0:
                    action_params[param_name] = self.np_random.choice(options)
                else:
                    invalid_actions.append(action_class)
                    action_params = None
                    break
            if action_params is not None:
                return chosen_host, action_class(**action_params)
            print("valid action but action params was None -> add it to invalid_actions to norrow options") #no prints


    def state_transitions_failure(self):
        """The state transition matrix for an unsuccessful action on a host.

        There is a row for each of the host states: K, KD, S, SD, U, UD, R, RD.
        Then there is a column for each of the actions, in the order of the `action_list`.
        
        All column 0 must have transition state as all hosts in subnet are transitioned

        ??? example
            ```
            map = {
                'K'  : ['K' , 'K' , 'K' , None, None, None, None, None, None],
                'KD' : ['KD', 'KD', 'KD', None, None, None, None, None, None],
                'S'  : ['S' , None, None, 'S' , 'S' , None, None, None, None],
                'SD' : ['SD', None, None, 'SD', 'SD', None, None, None, None],
                'U'  : ['U' , None, None, None, None, 'U' , None, None, 'U' ],
                'UD' : ['UD', None, None, None, None, 'UD', None, None, 'UD'],
                'R'  : ['R' , None, None, None, None, None, 'R' , 'R' , 'R' ],
                'RD' : ['RD', None, None, None, None, None, 'RD', 'RD', 'RD'],
                'F'  : ['F',  None, None, None, None, None, None, None, None],
            }
            ```

        Returns
        -------
        map : Dict[str, List[float]]
        """
        map = {
            'K'  : ['K' , 'K' , 'K' , None, None, None, None, None, None],
            'KD' : ['KD', 'KD', 'KD', None, None, None, None, None, None],
            'S'  : ['S' , None, None, 'S' , 'S' , None, None, None, None],
            'SD' : ['SD', None, None, 'SD', 'SD', None, None, None, None],
            'U'  : ['U' , None, None, None, None, 'U' , None, None, 'U' ],
            'UD' : ['UD', None, None, None, None, 'UD', None, None, 'UD'],
            'R'  : ['R' , None, None, None, None, None, 'R' , 'R' , 'R' ],
            'RD' : ['RD', None, None, None, None, None, 'RD', 'RD', 'RD'],
            'F'  : ['F',  None, None, None, None, None, None, None, None],
        }
        return map
