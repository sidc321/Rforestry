#!/bin/bash
set -e

echo === black ===
python3 -m black .  --exclude extension
echo

echo === isort ===
python3 -m isort . --color
echo

echo === flake8 ===
python3 -m flake8
echo

echo === bandit ===
python3 -m bandit -r .
echo

echo "It's all good to go!"