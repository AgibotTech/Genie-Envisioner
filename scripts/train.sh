#!/usr/bin/bash

script_path=${1}
config_path=${2}

# If WORLD_SIZE is not set, we are likely on a single node
if [ -z $WORLD_SIZE ]; then
    NGPU=`nvidia-smi --list-gpus | wc -l`
    echo "Training on 1 Node, $NGPU GPUs"
    
    # Check if MASTER_PORT is set in sbatch, otherwise default to 29505
    P_PORT=${MASTER_PORT:-29505}
    
    torchrun --nnodes=1 \
        --nproc_per_node=$NGPU \
        --node_rank=0 \
        --master_port=$P_PORT \
        $script_path \
        --config_file $config_path
else
    echo "Training on $WORLD_SIZE Nodes, $NGPU GPU per Node"
    NGPU=`nvidia-smi --list-gpus | wc -l`
    torchrun --nnodes=$WORLD_SIZE \
        --nproc_per_node=$NGPU \
        --node_rank=$RANK \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT \
        $script_path \
        --config_file $config_path
fi