import ray
from ray.rllib.algorithms.ppo import PPOConfig
import os
import socket
import time

hostname = socket.gethostname()

# Ray Initialization
ray.init()

print(f"--- Launching on node: {hostname} ---")
print(f"Detected Ray resources: {ray.cluster_resources()}")

config = (
    PPOConfig()
    .environment("Acrobot-v1")
    # Using 10 workers to leverage the high core count of G5K nodes (32+ cores)
    .env_runners(num_env_runners=10)
    .resources(num_gpus=0)
    .training(lr=0.0003)
)

start_time = time.time()
algo = config.build()

print("--- Starting Training ---")

for i in range(10):
    result = algo.train()
    
    runners_results = result.get("env_runners", {})
    mean_reward = runners_results.get("episode_reward_mean") or \
                  runners_results.get("episode_return_mean") or \
                  result.get("episode_reward_mean")

    if mean_reward is not None:
        print(f"Iteration {i+1} : Mean Score = {mean_reward}")
    else:
        print(f"Iteration {i+1} : Score pending...")

duration = time.time() - start_time
print("--- Finished ---")
print(f"Total time: {duration:.2f} seconds")

# Save Checkpoint
checkpoint_dir = algo.save().checkpoint.path
print(f"Model saved at: {checkpoint_dir}")

ray.shutdown()
