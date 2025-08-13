#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="localhost"

python3 web_infer_scripts/simple_client.py --host $IP_ADDRESS_OF_SERVER --port 8001 --env WM