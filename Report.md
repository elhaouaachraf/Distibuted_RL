# Parallel Programming for the AI Era: Distributed Reinforcement Learning Project

**Student:** Achraf Elhaoua  
**Professor:** Bruno Raffin  

---

## 1. Introduction

The objective of this project is to implement and deploy a distributed Reinforcement Learning (RL) workflow on a High-Performance Computing (HPC) cluster. Using the **Grid'5000** infrastructure, we train an agent on the `Acrobot-v1` environment using the **PPO** (Proximal Policy Optimization) algorithm.

The main goal is to master the software stack required for distributed AI (Ray, RLlib, OAR scheduler) and to analyze how parallelizing the data collection phase (workers) impacts training time and learning stability across multiple computing nodes.

---

## 2. Setup

### 2.1. Resource Contention and "Best Effort" Strategy

We encountered significant resource contention on the cluster. Standard reservation requests were rejected with long waiting times (over 8 hours):

```bash (base) aelhaoua@fgrenoble:~$ oarsub -I -l nodes=1,walltime=0:30 --project=lab-2025-uga-mosig-hpcai ```
```bash Filtering out exotic resources (servan, yeti, troll, drac, sasquatch). ```
```bash OAR_JOB_ID=2603241 ```
```bash Interactive mode: waiting... ```
```bash [2026-01-10 23:12:21] Start prediction: 2026-01-11 07:01:53 (FIFO scheduling OK) ```

---

### 2.2. Software Stack

- **Language:** Python 3.12 (Miniconda environment)  
- **Middleware:** Ray (cluster connectivity and actor management)  
- **Library:** RLlib (industry-grade Reinforcement Learning library)  
- **Environment:** Gymnasium `Acrobot-v1` (classic control problem)  

---

## 3. Experiments and Results

### 3.1. Baseline: Local Machine (Single Worker)

We first ran the training on a personal local machine to establish a baseline for stability and execution time using a single CPU.

**Configuration:**
- 1 Worker  
- 10 Iterations  

**Results:**
- **Time:** 132 seconds  
- **Best and Final Score:** -132  

**Observation:**  
The training converges to a good score (close to -100) with good stability.

---

### 3.2. Single Node on Grid'5000 (10 Workers)

To test the distributed setup, we reserved one node on Grid'5000.

**Node:** `chartreuse3-1.grenoble.grid5000.fr`  

**Ray detected resources:**  
`{'object_store_memory': 19824568320.0, 'memory': 46257326080.0, 'CPU': 32.0}`  

**Configuration:**
- 1 Node  
- 10 Workers  
- 10 Iterations  

**Results:**
- **Time:** 121 seconds  
- **Final and Best Score:** -301  

**Observations:**
- The same number of iterations is completed slightly faster than on the local machine.
- Despite using 10 workers, the result is worse than the single-worker local run. This is because, with many workers, the master receives a large amount of data early on. Paradoxically, having too much data at the beginning can slow down learning compared to a single worker focusing on a single experience stream.
- However, with a higher number of iterations, the 10-worker setup is expected to outperform the single-worker configuration.

To verify this hypothesis, we ran:
- 1 worker with 60 iterations on the local machine.
- 10 workers with 60 iterations on Grid'5000, using 2 nodes.

---

### 3.3. Local Machine (1 Worker, 60 Iterations)

**Configuration:**
- 1 Worker  
- 60 Iterations  

**Results:**
- **Time:** 709 seconds  
- **Final Score:** -113  
- **Best Score:** -111  

**Observation:**  
Good stability and good results, clearly better than the 10-iteration run.

---

### 3.4. Grid'5000 (2 Nodes, 10 Workers, 60 Iterations)

**Nodes:**
- **Head node:** `chartreuse3-2.grenoble.grid5000.fr`  
- **Worker node:** `vercors9-2.grenoble.grid5000.fr`  

**Total resources:**  
`{'memory': 93389535233.0, 'object_store_memory': 40024086527.0, 'CPU': 48.0}`  

**Configuration:**
- 2 Nodes  
- 10 Workers  
- 60 Iterations  

**Results:**
- **Time:** 624 seconds  
- **Final Score:** -108  
- **Best Score:** -99  

**Observation:**  
This configuration outperforms the local single-worker run both in terms of speed and final score, with good stability.

---

### 3.5. Max Performance Attempt (Dynamic Scaling)

In this experiment, we attempted to maximize cluster utilization by spawning one worker per available CPU core.

**Configuration:**
- 2 Nodes  
- Dynamic Workers (`total_cpus - 2 ≈ 46 workers`)  
- 60 Iterations  
- Same nodes as in the previous experiment  

**Results:**
- **Time:** 824 seconds  
- **Final Score:** -325  
- **Best Score:** -235  

**Analysis:**
- Contrary to expectations, maximizing the number of workers degraded both performance and learning quality.
- The execution time increased despite having the same number of iterations.

**Reasons:**
- **Communication overhead:** Managing around 46 workers created a bottleneck at the head node. For a lightweight environment like Acrobot, the time spent synchronizing observations and model weights over the network exceeded the gains from parallel simulation.
- **Convergence instability:** With 46 workers, the effective batch size became too large. Without hyperparameter tuning (especially the learning rate), the algorithm received too many averaged experiences at once, preventing effective convergence.

---

### 3.6. General Observations

Reinforcement learning is highly stochastic at the beginning. Sometimes, lucky initial trajectories allow the agent to quickly reach better scores, while in other cases it takes a long time to take off and reach stable improvement. In some runs, the training can remain stuck around -500 indefinitely, in which case restarting the program is necessary.

---

## 4. Conclusion

This project successfully demonstrated the deployment of a Ray cluster on the Grid'5000 infrastructure. The results highlight an important trade-off in distributed AI:

- **Distributed can be faster and more stable:** Using a moderate number of workers (10) on the cluster provided the best balance between execution time (624 s) and learning stability (best score of -99).
- **More is not always better:** Naively scaling to the maximum number of cores introduced significant communication overhead and algorithmic instability.

Future work could focus on tuning PPO hyperparameters (learning rate, batch size) to better support large-scale parallelism (40+ workers), or on testing heavier environments where the simulation cost justifies the communication overhead.
