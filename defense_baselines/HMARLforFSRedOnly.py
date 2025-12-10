import sys
import os
# Add CybORG parent directory to path
cyborg_path = '/projects/ImaniLab/Armita/CAGE-Multi-Adversary'
sys.path.insert(0, cyborg_path)

import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.WARN)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("gymnasium").setLevel(logging.ERROR)
logging.getLogger("ray.air").setLevel(logging.ERROR)
logging.getLogger("ray.tune.execution").setLevel(logging.ERROR)
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["RLIO_DISABLE_RLLIB_LOGGER"] = "1"

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from EnterpriseMAEHmarl import EnterpriseMAE
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPO
from hmarl_action_mask_model import TorchActionMaskModelHppo
import ray
from ray.train import RunConfig, CheckpointConfig
from ray.tune import Tuner, TuneConfig
from helper import parse_args
import gymnasium



# from ray.rllib.utils.annotations import override

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Based on https://github.com/adityavs14/Hierarchical-MARL/blob/main/h-marl-3policy/subpolicies/train_subpolicies.py

ModelCatalog.register_custom_model(
    "hmarl_model", TorchActionMaskModelHppo
)

# Number of blue agents and mapping to policy IDs
NUM_AGENTS = 5
POLICY_MAP = {}

for i in range(NUM_AGENTS):
    POLICY_MAP[f"blue_agent_{i}_master"]  = f"Agent{i}_master"
    POLICY_MAP[f"blue_agent_{i}_investigate"]  = f"Agent{i}_investigate"
    POLICY_MAP[f"blue_agent_{i}_recover"]  = f"Agent{i}_recover"




def env_creator_CC4(env_config: dict) -> MultiAgentEnv:
    """
    Instantiate the CybORG Enterprise scenario with a RANDOM red agent.
    """
 
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,    # ← RANDOMIZED red agent
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(env=cyborg)



# Register the environment with RLlib
register_env(name="CC4", env_creator=lambda config: env_creator_CC4(config))

# Policy mapping function
def policy_mapper(agent_id, episode, worker, **kwargs) -> str:
    """Map a CybORG agent ID to an RLlib policy ID."""
    return POLICY_MAP[agent_id]

# Build the PPO algorithm configuration
def build_algo_config():
    """
    Returns a configured PPOConfig for the CC4 multi-agent setup.
    """
    # Instantiate one env to retrieve spaces
    env = env_creator_CC4({})


    OBSERVATION_SPACE = {}
    ACTION_SPACE = {}

    for i in range(NUM_AGENTS):
        OBSERVATION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.Dict({'action_mask': gymnasium.spaces.multi_discrete.MultiDiscrete([2,2]),'observations':env.observation_space(f'blue_agent_{i}')['observations'], 'id':gymnasium.spaces.discrete.Discrete(1)})
        ACTION_SPACE[f"Agent{i}_master"] = gymnasium.spaces.discrete.Discrete(2)

        OBSERVATION_SPACE[f"Agent{i}_investigate"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_investigate']})
        ACTION_SPACE[f"Agent{i}_investigate"] = env.action_space(f"blue_agent_{i}")

        OBSERVATION_SPACE[f"Agent{i}_recover"] = gymnasium.spaces.Dict({'action_mask': env.observation_space(f"blue_agent_{i}")['action_mask'], 'observations':env.observation_space(f"blue_agent_{i}")['obs_recover']})
        ACTION_SPACE[f"Agent{i}_recover"] = env.action_space(f"blue_agent_{i}")


    config = (
        PPOConfig()
        .framework("torch")
        .debugging(log_level='DEBUG') 
        .environment(
            env="CC4",
            env_config={"horizon": 500, "soft_horizon": True},   # ← ADD THIS
        )
        .resources(
            num_gpus=1, # Use if GPUs are available
            num_cpus_per_worker=1,      # ← ADD THIS (fixes missing worker CPU allocation)

        )
        .env_runners(
            batch_mode="complete_episodes",
            # Original Values
            num_env_runners=31, # parallel sampling, set 0 for 
            num_cpus_per_env_runner=1,
            # sample_timeout_s=None, # time for each worker to sample timesteps
            # GPT suggested
            # num_env_runners=20,
            # num_cpus_per_worker=0.25,
            sample_timeout_s=600,                     # ← NEW (prevents infinite hangs)
           #  worker_health_timeout_s=60,           # TypeError: AlgorithmConfig.env_runners() got an unexpected keyword argument 'worker_health_timeout_s' 
        )
        .multi_agent(
            policies={
                ray_agent: PolicySpec(
                    policy_class = PPOTorchPolicy,
                    observation_space = OBSERVATION_SPACE[ray_agent],
                    action_space = ACTION_SPACE[ray_agent],
                    config = {"entropy_coeff": 0.001},
                )
                for ray_agent in OBSERVATION_SPACE
            },
            policy_mapping_fn=policy_mapper,
        )
       #  .fault_tolerance(
           #  fail_fast='none',                      # ← NE TypeError: AlgorithmConfig.fault_tolerance() got an unexpected keyword argument 'fail_fast'W
           #  max_failures=50,                         # ← NE TypeError: AlgorithmConfig.fault_tolerance() got an unexpected keyword argument 'max_failures'W
       #  )

        .training(
            model={"custom_model": "hmarl_model"},
            lr=1e-5,
            grad_clip_by="global_norm",
            grad_clip=0.2,
            # train_batch_size=150000,     # Original 
            # minibatch_size=6000,         # Original 
            train_batch_size=20000,      # ← MUCH safer
            minibatch_size=2000,         # ← scales with above
           #  rollout_fragment_length=64,  # ← add thi TypeError: AlgorithmConfig.training() got an unexpected keyword argument 'rollout_fragment_length's

            # ignore_worker_failures=True,                 # ← NEW TypeError: AlgorithmConfig.training() got an unexpected keyword argument 'ignore_worker_failures'
            # recreate_failed_workers=True,                # ← NEW TypeError: AlgorithmConfig.training() got an unexpected keyword argument 'recreate_failed_workers'
            # restart_failed_sub_environments=True,        # ← NEW TypeError: AlgorithmConfig.training() got an unexpected keyword argument 'restart_failed_sub_environments'
        )
        .experimental(
            _disable_preprocessor_api=True,  
        )
    )

    return config

def run_training(cluster):
    # ORIGINAL CODE
    # runtime_env = {
    # "env_vars": {
    #     "PYTHONPATH": "/projects/ImaniLab/Armita/CAGE-Multi-Adversary"
    # }
    # }
    
    # GPT Eddited 
    runtime_env = {
    "working_dir": "/projects/ImaniLab/Armita/CAGE-Multi-Adversary/CybORG",
    "env_vars": {
        "PYTHONPATH": "/projects/ImaniLab/Armita/CAGE-Multi-Adversary"
    }
        
}

    
    if cluster and not ray.is_initialized():
        # Connect to the cluster
        ray.init(address="auto", runtime_env=runtime_env)
    else:
        # For local Ray instance
        ray.init(runtime_env=runtime_env)

    config = build_algo_config()

    tuner = Tuner(
        PPO,                              
        param_space=config,
        tune_config=TuneConfig(
            num_samples=1, # how many Optuna trials. Each time with different sampling 
        ),
        run_config=RunConfig(
            name="HMARLforFSRedOnly",
            storage_path="/projects/ImaniLab/Armita/CAGE-Multi-Adversary/defense_baselines/HMARL",
            stop={"training_iteration": 400},
            # ADDEDD by CLAUDE AI 
            checkpoint_config=CheckpointConfig(
            checkpoint_frequency=1,  # Save every 1 iterations
            num_to_keep=3,  # Keep last 5 checkpoints
            checkpoint_at_end=True,
            # checkpoint_interval=600,            # ← NEW: save every 10 minutes too
    )
            # **({"resume_from_checkpoint": restore_path} if restore_path else {})   # to load from a checkpoint,
        ),
    )

    # to restore from an interrupted run 
    # tuner = Tuner.restore(
    # path="/projects/ImaniLab/Armita/CAGE-Multi-Adversary/defense_baselines/HMARL/PPO_2025-11-30_15-18-12",
    # trainable=PPO
    # )
    

    result_grid = tuner.fit()


if __name__ == "__main__":
    
    args = parse_args()     
    run_training(args.cluster)
