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

class FSRedAgentTargetB(FiniteStateRedAgent):
    
    def __init__(self, name=None, np_random=None, agent_subnets=None):
        
        super().__init__(name, np_random)
        self.step = 0
        self.action_params = None
        self.last_action = None
        self.host_states = {} #has the IPs as keys and 'hostnames' associated, and what 'state' each host is in.
        self.host_service_decoy_status = {} #ports associated to each IP
        self.agent_subnets = agent_subnets #was mostly static
        self.action_list = super().action_list()
        self.zone_b_keywords = ['restricted_zone_b', 'operational_zone_b']
        self.host_state_sequence = []
        self.print_action_output = False
        self.print_obs_output = False
        self.prioritise_servers = False

        self.host_states_priority_list = super().set_host_state_priority_list()
        self.state_transitions_success = super().state_transitions_success() #what to do when success
        self.state_transitions_failure = super().state_transitions_failure() #what to do when fail
        self.state_transitions_probability = super().state_transitions_probability()    
    def _is_zone_b(self, hostname: str) -> bool:
        if hostname is None:
            return False
        lower = hostname.lower()
        return any(k in lower for k in self.zone_b_keywords)

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
        bool = False
        if self.host_states_priority_list is None:
            # print("is On")
            state_host_options = host_options
            # print(f"host_options: {host_options}")
            available_hosts = {};
            for h_opt in host_options:
                # print(f"self.host_stats: {self.host_states}")
                state = self.host_states[h_opt]['state']
                weight = 1
    
                # --- Zone B boost ---
                hostname = self.host_states[h_opt].get('hostname')
                if self._is_zone_b(hostname) and state not in ('RD'):
                    bool = True
                    weight *= 100  # boost Zone B priority
                if hostname is None and state not in ('KD', 'SD'):
                    # print("hosstname None" + state)
                    weight *= 20  # boost Not-known hosts priority
    
                if not hostname in available_hosts.keys():
                    available_hosts[h_opt] = weight
                else:
                    # keep max weight if multiple hosts with same state
                    available_hosts[h_opt] = max(available_hosts[h_opt], weight)
    
                if len(available_hosts) == len(state_host_options):
                    break
                    
            # if bool:
                # print(f"available_hosts: {available_hosts}")
            base = sum(available_hosts.values())
            # print(base)
            probs = [(p/base) for p in available_hosts.values()]
            # if bool: 
                # print(f"probes: {probs}")
            chosen_host = self.np_random.choice(list(available_hosts.keys()), p=probs)
            # if bool:
                # print(self.host_states[chosen_host].get('hostname'))
            return chosen_host

        else:
            print("Prioritizing B Is not implemented here")
            base = 100
            available_states = {}
            
            # for h_opt in host_options:
            #     if not self.host_states[h_opt]['state'] in available_states.keys():
            #         available_states[self.host_states[h_opt]['state']] = self.host_states_priority_list[self.host_states[h_opt]['state']] 
            #     if len(available_states) == len(self.host_states_priority_list):
            #         break

            for h_opt in host_options:
                state = self.host_states[h_opt]['state']
                weight = self.host_states_priority_list[state]
    
                # --- Zone B boost ---
                hostname = self.host_states[h_opt].get('hostname')
                if self._is_zone_b(hostname): #and state not in ('RD'):
                    weight *= 10  # boost Zone B priority
                    
    
                if not state in available_states.keys():
                    available_states[state] = weight
                else:
                    # keep max weight if multiple hosts with same state
                    available_states[state] = max(available_states[state], weight)
    
                if len(available_states) == len(self.host_states_priority_list):
                    break

            
            if sum(available_states.values()) > 0:
                p_multiplier = 1/((sum(available_states.values()) / base))
                probs = [(p/base)*p_multiplier for p in available_states.values()]
                chosen_state = self.np_random.choice(list(available_states.keys()), p=probs)
            else:
                chosen_state = self.np_random.choice(list(available_states.keys()))

            state_host_options = [h for h in host_options if self.host_states[h]['state'] == chosen_state]

        #if attacker is set to have a priority over server hosts
        if self.prioritise_servers and len(state_host_options) > 1:
            server_state_host_options = [h for h in state_host_options if self.host_states[h]['hostname'] is not None and 'server' in self.host_states[h]['hostname']] 
            if len(server_state_host_options) > 0:
                i = self.np_random.random()
                if i <= 0.75:
                    chosen_host = self.np_random.choice(server_state_host_options)
                else:
                    #pick other host type
                    if not len(server_state_host_options) == len(state_host_options):
                        non_server_state_host_options = [h for h in state_host_options if not h in server_state_host_options]
                        chosen_host = self.np_random.choice(non_server_state_host_options)
                    else:
                        chosen_host = self.np_random.choice(server_state_host_options)
            else:
                chosen_host = self.np_random.choice(state_host_options)        
        #if attacker does not have a priority over server hosts
        else:
            chosen_host = self.np_random.choice(state_host_options)
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
                is_zone_b = self._is_zone_b(hostname)
                probabilities = []
                for opt in options:
                    name = opt.__name__.lower()
                    if is_zone_b:
                        # --- Zone B → exploitation focus ---
                        if "exploit" in name or "remote" in name:
                            host_action_options[opt] *= 4
    
                    elif hostname is None:
                        # --- Non-Zone B → discovery focus ---
                        if "exploit" in name:
                            host_action_options[opt] *= 1
                        if "stealth" in name:
                            host_action_options[opt] *= 2
                    else: 
                        if "withdraw" in name:
                            host_action_options[opt] += 0.1
                        if "escalate" in name:
                            host_action_options[opt] *= 0.5


                    probabilities.append(host_action_options[opt])
                    # print(host_action_options[opt]) # prints an integer: e.g. 0.5 | 0.25 
                if not is_zone_b:
                    # print(f"for host {self.host_states[chosen_host]} and actions {options} with probs {probabilities}")
                    llllll = 0
                action_class = self.np_random.choice(options, p=np.array(probabilities)/sum(probabilities))
            else:
                print("zone B host with no valid actions") #no prints
                new_options = host_options[:]
                new_options.pop(chosen_host)
                return self._choose_action(action_space, new_options)
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


