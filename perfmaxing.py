import ray
from ray.rllib.algorithms.ppo import PPOConfig
import os
import socket
import time

hostname = socket.gethostname()
print(f"--- Script started on HEAD NODE: {hostname} ---")

try:
    ray.init(address="auto", _node_ip_address="0.0.0.0")
except:
    ray.init()

# Dynamic Scaling: Use all available CPUs minus 2 (overhead)
total_cpus = int(ray.cluster_resources().get("CPU", 1))
nb_workers = max(1, total_cpus - 2) 

print(f"Total Cluster Resources: {ray.cluster_resources()}")
print(f"--- Launching with {nb_workers} WORKERS (Max Utilization) ---")

config = (
    PPOConfig()
    .environment("Acrobot-v1")
    .env_runners(num_env_runners=nb_workers)
    .resources(num_gpus=0)
    .training(lr=0.0003)
)

algo = config.build()

start_time = time.time()

for i in range(60):
    result = algo.train()
    
    runners_results = result.get("env_runners", {})
    mean_reward = runners_results.get("episode_reward_mean") or \
                  runners_results.get("episode_return_mean") or \
                  result.get("episode_reward_mean")
    
    if mean_reward is not None:
        print(f"Iteration {i+1} : Score = {mean_reward:.2f}")
    else:
        print(f"Iteration {i+1} : Score pending (synchronization)...")

duration = time.time() - start_time
print(f"--- Finished in {duration:.2f} seconds with {nb_workers} workers ---")

ray.shutdown()
