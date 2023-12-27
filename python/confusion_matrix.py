from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import click
import os

@click.command()
@click.argument('current_datetime', type=str, required=True)

def confusionmatrix(current_datetime):
    datetime_str = current_datetime
    
    LOADFROM = f"../training/{datetime_str}/eval/"
    MATRIX_PATH = f"../training/{datetime_str}/confusion_matrices/"

    all_labels = np.load(LOADFROM + "labels.npy")
    all_preds = np.load(LOADFROM + "preds.npy")
    labels = [1, 2, 3]

    if not os.path.exists(MATRIX_PATH):
        os.makedirs(MATRIX_PATH)

    # Standard Confusion Matrix
    create_matrix(all_labels, all_preds, labels, "Confusion Matrix", 0, 
                  MATRIX_PATH + "confusion_matrix.png")

    # Normalized by Rows
    create_matrix(all_labels, all_preds, labels, "Row-Normalized Confusion Matrix", 2, 
                  MATRIX_PATH + "row_normalized_confusion_matrix.png")

    # Normalized by Columns
    create_matrix(all_labels, all_preds, labels, "Column-Normalized Confusion Matrix", 1, 
                  MATRIX_PATH + "col_normalized_confusion_matrix.png")

    click.echo('Confusion Matrices Created')

def create_matrix(all_labels, all_preds, labels, title, normalize, filename):
    plt.figure()
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    if normalize == 1:  # Normalize by columns
        column_sums = cm.sum(axis=0)
        column_sums[column_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype('float') / column_sums[np.newaxis, :]
    elif normalize == 2:  # Normalize by rows
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]:.2f}' if normalize else f'{cm[i, j]:}',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    confusionmatrix()