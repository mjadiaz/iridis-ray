import ray
from ray import tune
from ray.tune.registry import register_env

import argparse

from pheno_game.envs.pheno_env import PhenoEnvContinuous_v0
from omegaconf import OmegaConf

env_config = OmegaConf.load('hep_tools.yaml')
select_env = 'PhenoEnvContinuous-v0'
register_env(select_env, lambda config: PhenoEnvContinuous_v0(env_config=env_config))

parser = argparse.ArgumentParser(description="Script for training RLlib agents")
parser.add_argument("--num-cpus", type=int, default=0) # 0 = Allocation of a single CPU core
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default=select_env)
parser.add_argument("--redis-password", type=str, default=None)
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--local-mode", action="store_true")
args = parser.parse_args()
print(args)

if args.redis_password is None:
	# Single node
	ray.init(local_mode=args.local_mode)
	num_cpus = args.num_cpus - 1
else:
	# On a cluster
	ray.init(_redis_password=args.redis_password, addres=os.environ["ip_head"])
	num_cpus = args.num_cpus - 1


tune.run(
	args.run,
	name=args.name_env,
	local_dir="/scratch/mjad1g20/ray_results/pheno_test_1",
	stop={"training_iteration":500},
	checkpoint_freq=2,
	config={
		"env": args.name_env,
		"num_workers": num_cpus,
		"num_gpus": args.num_gpus,
		"ignore_worker_failures": True,
		"framework": "torch",
		"train_batch_size": 128,
		"timesteps_per_iteration": 50,
		"use_lstm": True,
		}
	)
