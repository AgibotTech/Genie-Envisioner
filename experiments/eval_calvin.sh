#!/usr/bin/bash


python experiments/eval_calvin.py -s 0 -e 1000 -d 0 \
    -r evaluation_results/calvin \
    -c configs/ltx_model/calvin/action_model_calvin.yaml \
    -w path/to/trained/checkpoint



### 
### if you want to eval in parallel, use the codes bellow
# 
# n_gpu=8
# sidx=0
# step=$((1000/$n_gpu))
# for i in 0 1 2 3 4 5 6 7; do
# # echo  $(($i*$step)) $((($i+1)*$step))
# python experiments/eval_calvin.py -s $(($i*$step)) -e $((($i+1)*$step)) -d $i \
#     -r evaluation_results/calvin \
#     -c configs/ltx_model/calvin/action_model_calvin.yaml \
#     -w path/to/trained/checkpoint &
# done
# wait 
# 