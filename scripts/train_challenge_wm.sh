#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/cosmos_model/acwm_cosmos_challenge_wm_singleview.yaml}
SCRIPT_PATH=${2:-main.py}

bash "$(dirname "$0")/train.sh" "$SCRIPT_PATH" "$CONFIG_PATH"
