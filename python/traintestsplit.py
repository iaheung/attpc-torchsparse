import numpy as np
import os.path
from sklearn.model_selection import train_test_split
import click

@click.command()
@click.argument('loadfrom', type=str, required=True)
@click.argument('iso', type=str, required=True)

def traintestsplit(loadfrom, iso):
    # Define the ratio for the splits
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2
    
    # load data
    LOADFROM = loadfrom
    ISOTOPE = iso
    coords = np.load(LOADFROM + ISOTOPE + "_coords.npy")
    feats = np.load(LOADFROM + ISOTOPE + "_feats.npy")
    labels = np.load(LOADFROM + ISOTOPE + "_labels.npy")
    eventlens = np.load(LOADFROM + ISOTOPE + "_eventlens.npy")

    # Split the data into training, validation, and test sets
    coords_train, coords_temp, feats_train, feats_temp, labels_train, labels_temp, eventlens_train, eventlens_temp = train_test_split(coords, feats, labels, eventlens, test_size=1 - train_ratio, random_state=42)
    
    # Split the temporary sets into validation and test sets
    coords_val, coords_test, feats_val, feats_test, labels_val, labels_test, eventlens_val, eventlens_test = train_test_split(coords_temp, feats_temp, labels_temp, eventlens_temp, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)
    
    np.save(LOADFROM + ISOTOPE + "_coords_train.npy", coords_train)
    np.save(LOADFROM + ISOTOPE + "_coords_val.npy", coords_val)
    np.save(LOADFROM + ISOTOPE + "_coords_test.npy", coords_test)
    np.save(LOADFROM + ISOTOPE + "_feats_train.npy", feats_train)
    np.save(LOADFROM + ISOTOPE + "_feats_val.npy", feats_val)
    np.save(LOADFROM + ISOTOPE + "_feats_test.npy", feats_test)
    np.save(LOADFROM + ISOTOPE + "_labels_train.npy", labels_train)
    np.save(LOADFROM + ISOTOPE + "_labels_val.npy", labels_val)
    np.save(LOADFROM + ISOTOPE + "_labels_test.npy", labels_test)
    np.save(LOADFROM + ISOTOPE + "_eventlens_train.npy", eventlens_train)
    np.save(LOADFROM + ISOTOPE + "_eventlens_val.npy", eventlens_val)
    np.save(LOADFROM + ISOTOPE + "_eventlens_test.npy", eventlens_test)
    
    
    # Print shapes for validation data
    print("Training Coords Shape: ", coords_train.shape)
    print("Training Feats Shape: ", feats_train.shape)
    print("Training Labels Shape: ", labels_train.shape)
    print("Training Lengths Shape: ", eventlens_train.shape)
    print()
    # Print shapes for validation data
    print("Validation Coords Shape: ", coords_val.shape)
    print("Validation Feats Shape: ", feats_val.shape)
    print("Validation Labels Shape: ", labels_val.shape)
    print("Validation Lengths Shape: ", eventlens_val.shape)
    print()
    # Print shapes for test data
    print("Test Coords Shape: ", coords_test.shape)
    print("Test Feats Shape: ", feats_test.shape)
    print("Test Labels Shape: ", labels_test.shape)
    print("Test Lengths Shape: ", eventlens_test.shape)
    print()
    print("traintestsplit.py: Successful")

if __name__ == '__main__':
    traintestsplit()
