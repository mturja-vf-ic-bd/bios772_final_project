import numpy as np
import random
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def create_classification_pairs(x, y):
    pair_dict = {}
    idx_0_1 = np.concatenate((np.where(y == 0)[0], np.where(y == 1)[0]), axis=-1)
    idx_1_2 = np.concatenate((np.where(y == 1)[0], np.where(y == 2)[0]), axis=-1)
    idx_0_2 = np.concatenate((np.where(y == 0)[0], np.where(y == 2)[0]), axis=-1)
    pair_dict["NC_MCI"] = {"x" : x[idx_0_1], "y" : y[idx_0_1]}
    pair_dict["MCI_AD"] = {"x": x[idx_1_2], "y": y[idx_1_2] - 1}
    pair_dict["NC_AD"] = {"x": x[idx_0_2], "y": y[idx_0_2] // 2}
    return pair_dict


def create_data_loaders():
    x = np.load('../../data/train_x.npy')
    y = np.load('../../data/train_y.npy')
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x, y,
                                                      test_size=0.2,
                                                      random_state=SEED)
    data_dict_train = create_classification_pairs(x_train_split, y_train_split)
    for pair_name, data in data_dict_train.items():
        x_train, y_train = data["x"], data["y"]
        np.save(f"train_x_{pair_name}.npy", x_train)
        np.save(f"train_y_{pair_name}.npy", y_train)

    data_dict_val = create_classification_pairs(x_val_split, y_val_split)
    for pair_name, data in data_dict_val.items():
        x_val, y_val = data["x"], data["y"]
        np.save(f"val_x_{pair_name}.npy", x_val)
        np.save(f"val_y_{pair_name}.npy", y_val)
    np.save(f"val_x.npy", x_val_split)
    np.save(f"val_y.npy", y_val_split)


if __name__ == '__main__':
    create_data_loaders()
