#!/bin/bash
#SBATCH --job-name "installTorchSparse"
#SBATCH --mem 16g
#SBATCH --gpus 1

cd ../python
source /opt/conda/bin/activate sparse # change env name to your conda env

python install.py
