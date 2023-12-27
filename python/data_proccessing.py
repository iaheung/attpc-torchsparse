import h5py
import numpy as np
import os.path
import click

@click.command()
@click.argument('loadfrom', type=str, required=True)
@click.argument('iso', type=str, required=True)
@click.argument('h5', type=str, required=True)
#@click.argument('data_type', type=int, required=True) # 0 - point classification, 1 - track counting
    
def process_data(loadfrom, iso, h5):    
    LOADFROM = loadfrom
    H5 = h5
    file = h5py.File(LOADFROM + H5, 'r')
    
    original_keys = list(file.keys())
    original_length = len(original_keys)
    
    event_lens = np.zeros(original_length, int)
    for i in range(original_length):
        event = original_keys[i]
        event_lens[i] = len(file[event])
        
    ISOTOPE = iso
    file_name = ISOTOPE + '_w_key_index'
    
    # **only doing this if the file doens't exist already, as the conversion takes a while**
    if not os.path.exists(LOADFROM + file_name + '.npy'):
        event_data = np.zeros((original_length, np.max(event_lens), 13), float) 
        for n in range(len(original_keys)):
            name = original_keys[n]
            event = file[name]
            ev_len = len(event)
            #converting event into an array
            for i,e in enumerate(event):
                instant = np.array(list(e))
                event_data[n][i][:12] = np.array(instant)
                event_data[n][i][-1] = float(n) #insert index value to find corresponding event ID
        np.save(LOADFROM + file_name, event_data)
    
    data = np.load(LOADFROM + ISOTOPE + '_w_key_index' + '.npy')
        
    sliced_data = data[:, :, [0, 1, 2, 4, 5]]

    LENEVTS = len(sliced_data) # number of events (10000)
    LENDETS = len(sliced_data[0]) # number of detections (1476)
    NUMCLASSES = 5 # x, y, z, amp, track_id
    cutoff = 70 # discard events with less than 70 detections
    newLen = np.sum(event_lens > 70)
    
    new_data = np.zeros((newLen, LENDETS, NUMCLASSES), float)
    new_eventlens = np.zeros(newLen)
    new_data_index = 0
    
    print("Before Removing Unwanted Events - Length:", len(new_data))
    # point classification
    for i in range(LENEVTS):
        if event_lens[i] > 70:
            trackids = sliced_data[i][:, 4]
            # track_id 5 and 6 are not real events so only proccess events without id 5 and 6
            if not(np.any(trackids == 5) or np.any(trackids == 6)):
                # track_id 3 and 4 are both protons, so relabel id 4 to 3
                if (np.any(trackids == 4)):
                    sliced_data[i][trackids == 4, 4] = 3
                new_data[new_data_index] = sliced_data[i] 
                new_eventlens[new_data_index] = event_lens[i]
                new_data_index += 1
    
    # There are still events with no reaction even though we have largely filtered them out by using events
    # with more than 70 points, but we need to filter them out as we do not care about non-reaction events
    labels = new_data[:, :, -1]
    events_to_keep = []
    for i in range(len(new_data)):
        unique_labels = np.unique(labels[i])
        # Check if there are labels other than '0'
        keep_event = np.any(unique_labels != 0)
        events_to_keep.append(keep_event)

    events_to_keep = np.array(events_to_keep)
    new_data = new_data[events_to_keep]
    new_eventlens = new_eventlens[events_to_keep]
    print("After Removing Unwanted Events - Length:", len(new_data))
    print()
    
    # Since newLen is before we removed events with unwanted track ids, we need to trim the new data array
    non_zero_rows = np.any(new_data, axis=(1, 2))
    new_data = new_data[non_zero_rows]
    new_eventlens = new_eventlens[new_eventlens != 0]

    # mins and max
    # Min values for x, y, z, amp: [-250.32000732 -252.37495422  -56.]
    # Max values for x, y, z, amp: [  250.32003784   252.37495422   894.4]
    mins = new_data[:, :, :3].min(axis=(0, 1))
    maxs = new_data[:, :, :3].max(axis=(0, 1))

    # slice the new_data array so we can voxelize x, y, z
    x_slice, y_slice, z_slice, amp_slice, track_slice = [new_data[:, :, i:i+1] for i in range(5)]

    x_voxel = linearfit(x_slice, mins[0], maxs[0], 0, 500).astype(int)
    y_voxel = linearfit(y_slice, mins[1], maxs[1], 0, 500).astype(int)
    z_voxel = linearfit(z_slice, mins[2], maxs[2], 0, 900).astype(int)

    # z-score normalization on the amp
    non_zero_amp_slice = amp_slice[amp_slice != 0] # do not include 0s in the mean and std dev calculation
    mean_non_zero = non_zero_amp_slice.mean()
    std_dev_non_zero = non_zero_amp_slice.std()
    normalized_amp = (amp_slice - mean_non_zero) / std_dev_non_zero

    voxel_data = np.concatenate((x_voxel, y_voxel, z_voxel), axis=2)
    features = np.concatenate((voxel_data, normalized_amp), axis=2)
    
    np.save(LOADFROM + ISOTOPE + "_coords.npy", voxel_data)
    np.save(LOADFROM + ISOTOPE + "_feats.npy", features)
    np.save(LOADFROM + ISOTOPE + "_labels.npy", track_slice)
    np.save(LOADFROM + ISOTOPE + "_eventlens.npy", new_eventlens)

    print("Coords Shape: ", end="")
    print(voxel_data.shape)
    print("Feats Shape: ", end="")
    print(features.shape)
    print("Labels Shape: ", end="")
    print(track_slice.shape)
    
    print("data_processing.py: Successful")

def linearfit(x, min_orig, max_orig, min_desired, max_desired):
    """
    Perform linear scaling (fitting) of a value within a specified range to a desired range.

    Args:
        x (numpy.ndarray): The value to be scaled.
        min_orig (float): The minimum value of the original range.
        max_orig (float): The maximum value of the original range.
        min_desired (float): The minimum value of the desired range.
        max_desired (float): The maximum value of the desired range.

    Returns:
        numpy.ndarray: A NumPy array containing scaled values within the desired range.
    """
    linfit = np.copy(x)
    
    # Apply linear scaling only to non-zero elements
    non_zero_mask = x != 0
    linfit[non_zero_mask] = ((x[non_zero_mask] - min_orig) / (max_orig - min_orig)) * (max_desired - min_desired) + min_desired
    return linfit

if __name__ == '__main__':
    process_data()
