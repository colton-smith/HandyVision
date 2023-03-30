#!/bin/bash

echo "--- Creating python virtual environment ---"
python3 -m venv .venv

echo "--- Activating python virtual environment ---"
source .venv/bin/activate

echo "--- Installing python requirements ---"
pip install -r requirements.txt

echo "--- Installing handyvision package in development mode ---"
cd lib
pip install -e .

echo "--- Checking for existence of handyvision package ---"
pip freeze | grep handyvision
