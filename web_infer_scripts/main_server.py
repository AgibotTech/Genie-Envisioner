import os
import torch
import logging
import socket
import sys
import argparse
import json

root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from web_infer_utils.server import MVActorServer



def get_args():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML model config')

    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to the model weight')

    parser.add_argument('--host', type=str, required=True, help='IP address of server')

    parser.add_argument('-p', '--port', type=int, default=8001)

    parser.add_argument('--domain_name', type=str, default="agibotworld")

    parser.add_argument("--add_state", action="store_true")
    
    parser.add_argument('--threshold', type=float, default=200, help='The number of steps to update memories')

    parser.add_argument('--denoise_step', type=int, default=5)

    parser.add_argument('--action_dim', type=int, default=16)

    args = parser.parse_args()

    if args.threshold < 0:
        args.threshold = None

    return args



if __name__ == "__main__":

    """
    This script provides a simple way to build a web server of GEAct based on the serving codes in openpi_client (modified from https://github.com/Physical-Intelligence/openpi)
    """

    args = get_args()
    policy_metadata = dict(test_meta="Genie Envisioner Action Model")

    ### init actor
    actor = MVActorServer(
        args.host, args.port, policy_metadata,
        config_file=args.config,
        transformer_file=args.weight,
        load_weights=True,
        threshold=args.threshold,
        domain_name=args.domain_name,
        num_inference_steps=args.denoise_step,
        action_dim=args.action_dim,
        gripper_dim=1,
    )

    ### init server
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    print("Waiting...")

    ### start server and waiting for response
    actor.serve_forever()
