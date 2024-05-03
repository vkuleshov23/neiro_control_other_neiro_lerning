import csv
import os
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

trained = "trained"
understudied = "understudied"
retrained = "retrained"
batch_size = 64
AUTOTUNE = tf.data.AUTOTUNE


# def prepare_to_batch(ds):
#     ds = ds.batch(batch_size)
#     ds = ds.cache().prefetch(AUTOTUNE)
#     return ds


def get_shape(path):
    params = read_csv(path)
    return params.shape


def get_type(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2].numpy().decode('utf-8')


def read_types(directory):
    types = []
    for dir_path in pathlib.Path(directory).glob('*'):
        types.append(str(os.path.basename(dir_path)))
    return np.array(types)


def read_csv(filename):
    data = pd.read_csv(filename)
    params = []
    loss = data['loss'].values
    val_loss = data['val_loss'].values
    acc = data['acc'].values
    val_acc = data['val_acc'].values
    params = np.concatenate((params, loss))
    params = np.concatenate((params, val_loss))
    params = np.concatenate((params, acc))
    params = np.concatenate((params, val_acc))
    # params.append(np.array(loss))
    # params.append(np.array(val_loss))
    # params.append(np.array(acc))
    # params.append(np.array(val_acc))
    return np.array(params)


def shuffle_filenames(directory):
    directory = pathlib.Path(directory)
    names = []
    for file_path in directory.glob('**/*.csv'):
        names.append(str(file_path))
    np.random.shuffle(names)
    num_samples = len(names)
    print("Dataset size: ", num_samples)
    print("Random file: ", names[0])
    return names


def get_params(params_from_ds, type_from_ds):
    return params_from_ds


def get_params_and_type(file_path):
    type_ = get_type(file_path)
    params = read_csv(file_path)
    return params, type_


def split_files(files):
    train = files[:len(files) * 0.7]
    validation = files[(len(files) * 0.7): ((len(files) * 0.7) + (len(files) * 0.15))]
    test = files[-(len(files) * 0.15):]
    return train, validation, test


# def read_files(file_names):
#     files = []
#     for file_name in file_names:
#         files.append((read_csv(file_name), get_type(file_name)))
#     return np.array(files)


def get_ds(file_names, all_types):
    data = []
    types = []
    for name in file_names:
        x, y = get_params_and_type(name)
        data.append(x)
        types.append(np.argmax(y == all_types))
    data = np.array(data)
    types = np.array(types)
    return data, types


def get_dataset(path):
    types = read_types(path)
    file_names = shuffle_filenames(path)
    train_files, validation_files, test_files = split_files(file_names)
    train_ds_data, train_ds_label = get_ds(train_files, types)
    validation_ds_data, validation_ds_label = get_ds(validation_files, types)
    test_ds_data, test_ds_label = get_ds(test_files, types)
    return train_ds_data, train_ds_label, validation_ds_data, validation_ds_label, test_ds_data, test_ds_label, types
