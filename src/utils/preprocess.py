########################################################
# This script is for merging individual npy data       # 
# into a single numpy array and preprocessing the data #
# by changing missing/infinite values to 0 and         #
# standardize the data to mean 0 and sd 1.             #
########################################################
# Sept. 10, 2021 by Owen Jiang (owenjf@live.unc.edu) 


import os
import numpy as np
import pandas as pd


def load_data(subject_dir, csv_path):
    """
    subject_dir: directory to the folder that contains Subject_xxxx.npy data
    csv_path: directory to label.csv
    """
    df = pd.read_csv(csv_path, index_col=0)
    subjects = os.listdir(subject_dir)

    x = []
    y = []
    for subject in subjects:
        features_path = os.path.join(subject_dir, subject)
        if not os.path.exists(features_path) or not features_path.endswith('npy'):
            continue
        else:
            row = df.loc[df["new_subject_id"] == subject.split('.')[0]]
            label = int(row['Label'])

            x.append(np.load(features_path))
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    return x, y


def main():
    """
    The main function you are going to run.
    """
    
    train_x, train_y = load_data(r'../../data/training_data', r'../../data/train_label.csv')
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    train_x = (train_x - mean) / std # you may encounter warning, that's fine.
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0, neginf=0)
    np.save('../../data/train_x.npy', train_x)
    np.save('../../data/train_y.npy', train_y)

    """
    test_x, test_y = load_data(r'./test_data', r'./label.csv')
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0, neginf=0)
    test_x = (test_x - mean) / std # you may encounter warning, that's fine.
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0, neginf=0)
    np.save('test_x.npy', test_x)
    np.save('test_y.npy', test_y)
    """


if __name__=='__main__':
    main()


