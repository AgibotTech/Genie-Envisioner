#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="localhost"
DOMAIN_NAME="DATASETNAME"

python3 web_infer_scripts/main_server.py \
    -c configs/ltx_model/policy_model.yaml \
    -w path/to/trained/checkpoint/diffusion_pytorch_model.safetensors \
    --add_state \
    --denoise_step 10 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --domain_name ${DOMAIN_NAME}
