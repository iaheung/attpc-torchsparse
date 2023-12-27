#!/bin/bash
#SBATCH --job-name "Training"
#SBATCH --mem 64g
#SBATCH --gpus 1

current_datetime="$(date "+%Y-%m-%d-%H:%M:%S")"

loadfrom="../mg22simulated/" # where the data is stored
iso="Mg22" # isotope of the data

learning_rate=0.000001
epochs=1000
batch_size=12

echo "Current datetime: $current_datetime"
echo "Load from: $loadfrom"
echo "Isotope: $iso"
echo "Learning rate: $learning_rate"
echo "Epochs: $epochs"
echo "Batch size: $batch_size"
echo

cd ../python
source /opt/conda/bin/activate sparse # change env name to your conda env

echo "Training Model"
python training.py $current_datetime $loadfrom $iso $learning_rate $epochs $batch_size
echo
echo "Generating Loss Curves"
python plotting.py $current_datetime $epochs
echo
echo "Evaluating Model"
python evaluate.py $current_datetime $loadfrom $iso $learning_rate $epochs $batch_size
echo
echo "Generating Confusion Matrices"
python confusion_matrix.py $current_datetime