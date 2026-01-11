#!/bin/bash

# 1. Activate Python Environment
source $HOME/miniconda3/bin/activate

# 2. Retrieve the list of reserved nodes
# We keep only one line per unique machine
uniq $OAR_NODEFILE > /tmp/nodes_list

# The first node acts as the HEAD
HEAD_NODE=$(head -n 1 /tmp/nodes_list)

# The remaining nodes act as WORKERS
WORKER_NODES=$(tail -n +2 /tmp/nodes_list)

echo "Head Node: $HEAD_NODE"
echo "Worker Nodes: $WORKER_NODES"

# 3. Start Ray on the HEAD node
echo "Starting Ray on Head..."
ssh $HEAD_NODE "source $HOME/miniconda3/bin/activate && ray stop && ray start --head --port=6379 --disable-usage-stats"

# 4. Start Ray on the WORKER nodes
# We loop through workers to connect them to the Head
for worker in $WORKER_NODES; do
    echo "Connecting worker $worker to Head..."
    ssh $worker "source $HOME/miniconda3/bin/activate && ray stop && ray start --address=$HEAD_NODE:6379 --disable-usage-stats"
done

# Wait for connections to stabilize
sleep 5

# 5. Launch the Python script on the HEAD node
echo "Launching Python script..."
ssh $HEAD_NODE "source $HOME/miniconda3/bin/activate && python $HOME/projet_acrobot.py"

# 6. Cleanup (Stop Ray)
echo "Cleaning up..."
ssh $HEAD_NODE "ray stop"
for worker in $WORKER_NODES; do
    ssh $worker "ray stop"
done
