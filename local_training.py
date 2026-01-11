import ray
from ray.rllib.algorithms.ppo import PPOConfig
import os
import time 

# 1. Initialization
ray.init()

# 2. Configuration
config = (
    PPOConfig()
    .environment("Acrobot-v1")
    # Using 1 worker for local baseline
    .env_runners(num_env_runners=1)
    .resources(num_gpus=0)
    .training(lr=0.0003)
)

start_time = time.time()

# 3. Build Algorithm
algo = config.build()

print("--- Starting training on Acrobot (Local) ---")
print("Goal: Swing up the robot to touch the line.")
print("Score range: -500 (fail) to 0 (perfect). Target: > -100.")

# 4. Training Loop
for i in range(10):
    result = algo.train()
    
    runners_results = result.get("env_runners", {})
    
    mean_reward = runners_results.get("episode_reward_mean") or \
                  runners_results.get("episode_return_mean") or \
                  result.get("episode_reward_mean") or \
                  result.get("episode_return_mean")
                  
    if mean_reward is not None:
        print(f"Iteration {i+1} : Mean Score = {mean_reward:.2f}")
    else:
        print(f"Iteration {i+1} : Score pending (episode in progress...)")

duration = time.time() - start_time
print("--- Training finished ---")
print(f"Duration: {duration:.2f} seconds")

# 5. Save Checkpoint
checkpoint_dir = algo.save().checkpoint.path
print(f"Model saved at: {checkpoint_dir}")

ray.shutdown()
