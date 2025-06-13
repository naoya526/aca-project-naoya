#!/bin/bash
apt-get update
apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
apt-get install -y git
git clone https://github.com/naoya526/aca-project-naoya
