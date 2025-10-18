#!/usr/bin/env bash
set -euo pipefail

python -m src.digest --config configs/config.yaml
python -m src.metrics --config configs/config.yaml
python -m src.summary_stub --config configs/config.yaml
