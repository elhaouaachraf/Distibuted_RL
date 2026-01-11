import ray
from ray.rllib.algorithms.ppo import PPOConfig
import os
import socket
import time

hostname = socket.gethostname()
print(f"--- Script started on HEAD NODE: {hostname} ---")

# CLUSTER INITIALIZATION
try:
    # Connection to the existing Ray cluster 
    # created by the OAR/Bash script across the 2 nodes.
    ray.init(address="auto", _node_ip_address="0.0.0.0")
    print("Successfully connected to existing Ray cluster!")
except:
    print("No cluster detected, falling back to local mode.")
    ray.init()

print(f"Total Cluster Resources: {ray.cluster_resources()}")

nb_workers = 10

config = (
    PPOConfig()
    .environment("Acrobot-v1")
    .env_runners(num_env_runners=nb_workers)
    .resources(num_gpus=0)
    .training(lr=0.0003)
)

print(f"--- Launching training with {nb_workers} workers for 60 iterations ---")
start_time = time.time()

algo = config.build()

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
print(f"--- Finished in {duration:.2f} seconds on {len(ray.nodes())} nodes ---")

# Checkpoint
checkpoint_dir = algo.save().checkpoint.path
print(f"Model saved at: {checkpoint_dir}")

ray.shutdown()
