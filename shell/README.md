# Shell

## Usage

There are three shell files, each with a different function. Before running any of them, change the conda enviroment name *sparse* to your conda enviroment name. 

## dataprocessing.sh

Loads the .h5 file and transforms the data for model training. 

This is the shell file that takes the raw data and turns it into a format ready for training by first extracting the essential data from the .h5 file (x, y, z, amp, track_id), and then organizing the data in the format the TorchSparse package specifies (Coords, Feats, Labels). Then a train test split is performed on the new data in a seperate python script.

Expected Behaviour (Dimensions may vary based on the data): 

```
Before Removing Unwanted Events - Length: 4591
After Removing Unwanted Events - Length: 4564

Coords Shape: (4564, 1476, 3)
Feats Shape: (4564, 1476, 4)
Labels Shape: (4564, 1476, 1)
data_processing.py: Successful

Training Coords Shape:  (2738, 1476, 3)
Training Feats Shape:  (2738, 1476, 4)
Training Labels Shape:  (2738, 1476, 1)
Training Lengths Shape:  (2738,)

Validation Coords Shape:  (913, 1476, 3)
Validation Feats Shape:  (913, 1476, 4)
Validation Labels Shape:  (913, 1476, 1)
Validation Lengths Shape:  (913,)

Test Coords Shape:  (913, 1476, 3)
Test Feats Shape:  (913, 1476, 4)
Test Labels Shape:  (913, 1476, 1)
Test Lengths Shape:  (913,)

traintestsplit.py: Successful
```

## training.sh

The training-evaluation pipeline. Runs 4 python scripts for training, loss curve plotting, model evaluation, and confusion matrix generation.

This is the shell file responsible for the whole training process. 

