import os
import shutil

chkpt_root = "tmp/test2"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

#ray_results = "{}/ray_results/".format(os.getenv("HOME")) 
ray_results = "/scratch/mjad1g20/PhenoGame/ray/ddpg/ray_results"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

import ray
ray.init(ignore_reinit_error=True)# local_mode=True)

from ray.tune.registry import register_env
from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0
from omegaconf import OmegaConf

env_config = OmegaConf.load('hep_tools.yaml')
select_env = 'PhenoEnvContiuous-v0'
register_env(select_env, lambda config: PhenoEnvContinuous_v0(env_config=env_config ))

import ray.rllib.agents.ddpg as ddpg
config = ddpg.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
config["framework"] = "torch"
config['replay_buffer_config']['capacity'] = 1000000
config["train_batch_size"] = 64
config["prioritized_replay"] = False
config["learning_starts"] = 65
agent = ddpg.DDPGTrainer(config, env=select_env)


status = "{:2d} reward {:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 5

for n in range(n_iter):
	result = agent.train()
	chkpt_file = agent.save(chkpt_root)
	print(status.format(
		n+1,
		result["episode_reward_min"],
		result["episode_reward_mean"],
		result["episode_reward_max"],
		result["episode_len_mean"],
		chkpt_file
		))
	
