import pandas as pd
import matplotlib.pyplot as plt
import random
import os, sys
import shutil
from scipy.signal import savgol_filter
import time
import argparse
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ipdb
from torch.utils.data import DataLoader, Dataset
import torch
torch.manual_seed(0)
random.seed(5)
# set seeds separately by packages torch, random and numpy. import order matters
import numpy as np
np.random.seed(0)

from conf import config
from dataset import VDBProposedDataset
from display import plot_and_accept
# Directory containing the Parquet files
parquet_directory = 'datafiltered/mapestimationfiltered'
# Directory to save the accepted files
accepted_directory = 'train_data/'
# File to store the accepted arrays
training_file = os.path.join(accepted_directory, 'training_data.parquet')

# Custom exception for handling clean exit
class SafeExit(Exception):
    """Custom exception for handling clean exit"""
    pass


# Function to plot data and get user input

# Main script
def main():
    files_list = os.listdir(parquet_directory)
    train_filenames, valid_filenames = train_test_split(files_list, test_size=0.2, random_state=42)

    train_dataset = VDBProposedDataset(
        parquet_directory, 
        train_filenames, 
        fs=125, 
        model_fs=125, 
        segments_duration=300,
        standardize_params=config["feat_standard"]
        )

    # Iterate over each file
    waves_array = np.empty((0, 37500, 2))
    nums_array = np.empty((0, 300, 4))
    for j in range(193, len(train_dataset)):
        waves, nums, mbps, cuffs = train_dataset[j]
        for i in range(0, 18, 4):
            if i < len(waves):
                wave = waves[i, :, :]
                num = nums[i, :, :]
                mbp = mbps[i, :]
                cuff = cuffs[i, :]
                # Plot data and get user input
                result_string = plot_and_accept(wave, num, mbp, cuff, j, i)
                if result_string == "accept":
                    # Append the accepted array to the training data
                    print(np.expand_dims(wave, axis=0).shape, waves_array.shape)
                    waves_array = np.concatenate((waves_array, np.expand_dims(wave, axis=0)), axis=0)
                    num = np.concatenate((num, np.expand_dims(mbp, axis=1), np.expand_dims(cuff, axis=1)), axis=1)
                    print(num.shape, nums_array.shape)
                    nums_array = np.concatenate((nums_array, np.expand_dims(num, axis=0)), axis=0)
                    print(f"Segment {str(i)} of patient {str(j)} accepted.")
                elif result_string == "reject":
                    print(f"Segment {str(i)} of patient {str(j)} rejected.")
                elif result_string == "save":
                    print("saving data.....")
                    np.savez("labelled_filtereddatabis.npz", nums=nums_array, waves=waves_array)
                    sys.exit(0)

if __name__ == '__main__':
    main()
