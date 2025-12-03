from inspect import signature
from typing import Union, List, Dict
from pprint import pprint
from ipaddress import IPv4Address
from numpy import invert

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Simulator.Actions.AbstractActions import DiscoverRemoteSystems, PrivilegeEscalate, Impact, DegradeServices, AggressiveServiceDiscovery, StealthServiceDiscovery, DiscoverDeception
from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import PIDSelectiveExploitActionSelector, ExploitRemoteService
from CybORG.Simulator.Actions.ConcreteActions.RedSessionCheck import RedSessionCheck
from CybORG.Simulator.Actions.ConcreteActions.Withdraw import Withdraw
from CybORG.Simulator.Actions import Sleep, Action, InvalidAction
import numpy as np


class FSRedAgentCombined(BaseAgent):


    def __init__(self, name=None, np_random=None, agent_subnets=None):
        super().__init__(name, np_random)
        self.step = 0
        self.action_params = None
        self.last_action = None
        self.last_host = None
        self.host_states = {} #has the IPs as keys and 'hostnames' associated, and what 'state' each host is in.
        self.host_service_decoy_status = {} #ports associated to each IP
        self.agent_subnets = agent_subnets #was mostly static
        self.action_list = self.action_list()

        self.print_action_output = False
        self.print_obs_output = False
        self.prioritise_servers = False

        self.host_states_priority_list = self.set_host_state_priority_list()
        self.state_transitions_success = self.state_transitions_success() #what to do when success
        self.state_transitions_failure = self.state_transitions_failure() #what to do when fail
        self.state_transitions_probability = self.state_transitions_probability()    

        """ added/modified parts """
        # Track deception discovery failures per host
        self.deception_failures = {}  # host -> failure count
        self.action_failures = {}  # Track action-specific failures

    
    def get_action(self, observation: dict, action_space):
     
        action = None
        success = None

        if 'success' in observation.keys():
            success = observation.pop('success')
        # print(success) is TRUE, FALSE, ...
        if 'action' in observation.keys():
            action = observation.pop('action')
        # print(action) is looks like Action Name plus hostname
        """ added part"""
        # Track action-specific failures per host
        if success.name == "FALSE" and self.last_host:
            # Initialize tracking structures if needed
            if not hasattr(self, 'action_failures'):
                self.action_failures = {}  # {host: {action_type: {'attempts': 0, 'failures': 0}}}
            
            # Get action type
            action_type = type(action).__name__ if action else 'Unknown'
            
            # Initialize host entry if needed
            if self.last_host not in self.action_failures:
                self.action_failures[self.last_host] = {}
            
            # Initialize action entry if needed
            if action_type not in self.action_failures[self.last_host]:
                self.action_failures[self.last_host][action_type] = {'attempts': 0, 'failures': 0}
            
            # Update counters
            self.action_failures[self.last_host][action_type]['attempts'] += 1
            self.action_failures[self.last_host][action_type]['failures'] += 1
            
            # Also track overall failures for backward compatibility
            self.deception_failures[self.last_host] = self.deception_failures.get(self.last_host, 0) + 1
        
        # Track successful actions too
        elif success.name == "TRUE" and self.last_host:
            if not hasattr(self, 'action_failures'):
                self.action_failures = {}
            
            action_type = type(action).__name__ if action else 'Unknown'
            
            if self.last_host not in self.action_failures:
                self.action_failures[self.last_host] = {}
            
            if action_type not in self.action_failures[self.last_host]:
                self.action_failures[self.last_host][action_type] = {'attempts': 0, 'failures': 0}
            
            # Only increment attempts for success (not failures)
            self.action_failures[self.last_host][action_type]['attempts'] += 1
            # print(self.action_failures)
        """ end """
            
        self._host_state_transition(action, success)
        self._process_new_observations(observation)
        self._session_removal_state_change(observation)

        if self.print_action_output:
            self.last_turn_summary(observation, action, success)

        if success.name == 'IN_PROGRESS':
            self.step += 1
            return Sleep()
        else:
            known_hosts = [h for h in self.host_states.keys() if not self.host_states[h]['state'] == 'F']

            chosen_host, action = self._choose_host_and_action(action_space, known_hosts)

            if isinstance(action, ExploitRemoteService) and chosen_host in list(self.host_service_decoy_status.keys()):
                action.exploit_action_selector = PIDSelectiveExploitActionSelector(excluded_pids=self.host_service_decoy_status[chosen_host])

            self.step += 1
            self.last_action = action
            return action

    def _host_state_transition(self, action: Action, success):
        if not action == None and not success.name == 'IN_PROGRESS':
            action_index = None
            action_type = [A for A in self.action_list if isinstance(action, A)]

            if len(action_type) == 1:
                action_index = self.action_list.index(action_type[0])
                action_params = signature(action_type[0]).parameters
                
                host_ips = []
                if 'ip_address' in action_params:
                    host_ips.append(str(action.ip_address))
                elif 'hostname' in action_params:
                    for ip, host_dict in self.host_states.items():
                        if host_dict['hostname'] == action.hostname:
                            host_ips.append(ip)
                            break
                elif 'subnet' in action_params:
                    for ip in self.host_states.keys():
                        if IPv4Address(ip) in action.subnet:
                            host_ips.append(ip)
            
                for host_ip in host_ips:
                    if host_ip in self.host_states.keys():
                        curr_state = self.host_states[host_ip]['state']
                        next_state = None
                        if success.value == 1:
                            next_state = self.state_transitions_success[curr_state][action_index]
                        else:
                            next_state = self.state_transitions_failure[curr_state][action_index]

                        if next_state == 'U':
                            next_state = 'F'
                            for a_subnet in self.agent_subnets:
                                if IPv4Address(host_ip) in a_subnet:
                                    next_state = 'U'

                        if next_state == None:
                            # i.e. if something happens that causes the host to be in a state where they cannot perform that action 
                            # (e.g. session removed during action duration, or error), then just use their previous state. 
                            next_state = curr_state
                            
                        self.host_states[host_ip]['state'] = next_state

    def _session_removal_state_change(self, observation):
        """The changing of state of hosts, where its session has been removed (by Blue)."""
        removed_hosts = []

        for ip, hs in self.host_states.items():
            if 'U' in hs['state'] or 'R' in hs['state']:
                removed_hosts.append(ip)

        for host, obs in observation.items():
            if host == 'message':
                continue

            if 'Sessions' in obs.keys():
                for i, sess in enumerate(obs['Sessions']):
                    host_ip = str(obs['Interface'][0]['ip_address'])
                    if host_ip in removed_hosts:
                        removed_hosts.remove(host_ip)
        
        for ip in removed_hosts:
            self.host_states[ip]['state'] = 'KD'

    def _process_new_observations(self, observation: dict):
        """The finding of new hosts in the past observation, and the discovery of any decoys."""
        # Update knowledge of new hosts and decoys
        for host_id, host_details in observation.items():
            hostname = None
            ip = None

            if host_id == 'message':
                continue
                
            # Identify hostname in obs
            if '_' in host_id:
                hostname = host_id
            elif 'System info' in host_details:
                if 'Hostname' in host_details['System info']:
                    hostname = host_details['System info']['Hostname']
            
            # Identify ip in obs
            if '.' in host_id:
                ip = host_id
            elif 'Interface' in host_details:
                ip = str(host_details['Interface'][0]['ip_address'])
            
            # If hostname already in host_states, identify ip
            if ip == None and not hostname == None:
                for h_ip, h_details in self.host_states.items():
                    if h_details['hostname'] == hostname:
                        ip = h_ip
                        break

            # set new host starting state
            host_state = {}
            if self.step == 0:
                host_state['state'] = 'U'
                if self.agent_subnets == None:
                    for sub_dict in host_details['Interface']:
                        if 'Subnet' in sub_dict.keys():
                            self.agent_subnets = [sub_dict['Subnet']]
                            break
            else:
                host_state['state'] = 'K'

            # if new ip info
            if not ip in self.host_states.keys():
                self.host_states[ip] = host_state
                self.host_states[ip]['hostname'] = hostname

            # if new hostname info
            if not ip == None and not hostname == None:
                if self.host_states[ip]['hostname'] == None:
                    self.host_states[ip]['hostname'] = hostname
            
            # if new decoy info
            if 'Processes' in host_details.keys():
                for process in host_details['Processes']:
                    if 'Properties' in process and 'PID' in process:
                        if 'decoy' in process['Properties']:
                            if host_id in self.host_service_decoy_status:
                                self.host_service_decoy_status[host_id].append(process['PID'])
                            else:
                                self.host_service_decoy_status[host_id] = [process['PID']]
                        
    def _choose_host(self, host_options: List[str]):
        """Weighted host selection based on deception failures."""
        
        # Calculate weights based on deception failures
        weights = []
        for host in host_options:
            failures = self.deception_failures.get(host, 0)
            # Exponentially decrease weight based on failures
            # Weight = 1.0 / (1.5 ^ failures), minimum weight of 0.1
            weight = max(0.1, 1.0 / (1.5 ** failures))
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        # print(weights)
        # Choose host using weighted selection
        chosen_host = self.np_random.choice(host_options, p=weights)

        return chosen_host
    
    
    def _choose_host_and_action(self, action_space: dict, host_options: List[str]):
        """The selection of a valid host and action to execute this step."""
        chosen_host = self._choose_host(host_options)
        self.last_host = chosen_host
        if chosen_host == None:
            return Sleep()

        host_action_options = {self.action_list[i]: prob for i, prob in enumerate(self.state_transitions_probability[self.host_states[chosen_host]['state']]) if not prob == None}

        invalid_actions = []
        while True:
            options = [i for i, v in action_space['action'].items() if v and i not in invalid_actions and i in list(host_action_options.keys())]
            if len(options) > 0:
                probabilities = []

                for opt in options:
                    state = self.host_states.get(chosen_host, {}).get('state', 'K')
                    
                    # Get action-specific failure rate
                    action_name = opt.__name__
                    
                    # Calculate success rate for this action on this host
                    if hasattr(self, 'action_failures') and chosen_host in self.action_failures:
                        action_stats = self.action_failures[chosen_host].get(action_name, {'attempts': 0, 'failures': 0})
                        
                        if action_stats['attempts'] > 0:
                            failure_rate = action_stats['failures'] / action_stats['attempts']
                            success_rate = 1 - failure_rate
                            
                            # Dynamically adjust probability based on success rate
                            # Actions with higher success rates get boosted, failures get reduced
                            if success_rate < 0.3:  # High failure rate (>70%)
                                # Severely reduce probability
                                host_action_options[opt] *= 0.1
                            elif success_rate < 0.5:  # Moderate failure rate
                                # Moderately reduce probability
                                host_action_options[opt] *= 0.5
                            elif success_rate > 0.7:  # High success rate
                                # Boost probability
                                host_action_options[opt] *= 1.5
                            # else: keep original probability for 50-70% success rate
                    
                    # Special handling for specific states and actions
                    if state in ['S', 'SD']:
                        # If DiscoverDeception has failed too many times, focus on exploitation
                        if self.deception_failures.get(chosen_host, 0) >= 3:
                            name = action_name.lower()
                            
                            # Boost exploitation actions when deception detection has failed
                            if "exploit" in name:
                                host_action_options[opt] *= 2.0  # Double the probability
                            elif "discover" in name and "deception" in name:
                                host_action_options[opt] *= 0.1  # Reduce deception discovery attempts
                    
                    # Ensure probabilities don't go to zero (maintain minimum exploration)
                    if host_action_options[opt] < 0.01:
                        host_action_options[opt] = 0.01
            
                        
                    probabilities.append(host_action_options[opt])
                action_class = self.np_random.choice(options, p=np.array(probabilities)/sum(probabilities))
            else:
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
            print("This shouldn't happen")
    
    def train(self, results):
        pass

    def end_episode(self): 
        pass

    def set_initial_values(self, action_space, observation):
        if type(action_space) is dict:
            self.action_params = {action_class: signature(action_class).parameters for action_class in action_space['action'].keys()}

    def last_turn_summary(self, observation: dict, action: str, success):
        action_str = None
        if not action == None:
            action_str = str(action)
        elif success.name == 'IN_PROGRESS':
            action_str = str(self.last_action)
        else: 
            action_str = "Initial Observation"

        print(f'\n** Turn {self.step} for {self.name} **')
        print(f"Action: {action_str}")
        print("Action Success: " + str(success))

        if self.print_obs_output:
            print("\nObservation:")
            pprint(observation)
            print("Host States:")
            pprint(self.host_states)
        
        if isinstance(observation.get('action'), InvalidAction):
            pprint(observation['action'].error)
    
    def action_list(self):
        actions = [
            DiscoverRemoteSystems,          #0
            AggressiveServiceDiscovery,     #1
            StealthServiceDiscovery,        #2
            DiscoverDeception,              #3
            ExploitRemoteService,           #4
            PrivilegeEscalate,              #5
            Impact,                         #6
            DegradeServices,                #7
            Withdraw                        #8
        ]
        return actions

    def state_transitions_success(self):
    
        map = {
            'K'  : ['KD', 'S',  'S',  None, None, None, None, None, None],
            'KD' : ['KD', 'SD', 'SD',  None, None, None, None, None, None],
            'S'  : ['SD', None, None, 'S' , 'U' , None, None, None, None],
            'SD' : ['SD', None, None, 'SD', 'UD', None, None, None, None],
            'U'  : ['UD', None, None, None, None, 'R' , None, None, 'S' ],
            'UD' : ['UD', None, None, None, None, 'RD', None, None, 'SD'],
            'R'  : ['RD', None, None, None, None, None, 'R' , 'R' , 'S' ],
            'RD' : ['RD', None, None, None, None, None, 'RD', 'RD', 'SD'],
            'F'  : ['F',  None, None, None, None, None, None, None, None],
        }
        return map

    def state_transitions_failure(self):
       
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

    def set_host_state_priority_list(self):
        return None


    def state_transitions_probability(self):
        """Modified probability matrix with DiscoverDeception consideration."""
        # Add DiscoverDeception as potential action for S and SD states
        map = {
            'K'  : [0.5,  0.25, 0.25, None, None, None, None, None, None],
            'KD' : [None, 0.5,  0.5,  None, None, None, None, None, None],
            'S'  : [0.25, None, None, 0.25, 0.5, None, None, None, None],
            'SD' : [None, None, None, 0.25, 0.75, None, None, None, None],
            'U'  : [0.5 , None, None, None, None, 0.5 , None, None, 0.0 ],
            'UD' : [None, None, None, None, None, 1.0, None, None, 0.0 ],
            'R'  : [0.5,  None, None, None, None, None, 0.25, 0.25, 0.0 ],
            'RD' : [None, None, None, None, None, None, 0.5,  0.5,  0.0 ],
        }
        return map