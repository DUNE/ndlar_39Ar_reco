#!/usr/bin/env bash

set -o errexit

module load python
echo "Setting up environment..."
python -m venv flow.venv
source flow.venv/bin/activate
pip install --upgrade pip setuptools wheel

# install h5flow
git clone https://github.com/larpix/h5flow.git
cd h5flow
#git checkout d2864e84d1bcd2553139373d2f39baab7a9bdfdf
pip install .
cd ..

# install ndlar_flow
git clone https://github.com/larpix/ndlar_flow
cd ndlar_flow
git checkout develop
pip install .
cd ..
