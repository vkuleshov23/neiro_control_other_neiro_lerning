import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import generate
import prep_dataset as prep

DATASET_PATH = ""
data_directory = pathlib.Path(DATASET_PATH)
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 100


def create_model(ds, types):
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(data=ds)
    print(ds[0].shape)
    print(types)
    model_ = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=ds[0].shape),
        norm_layer,
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(types)),
    ])
    model_.summary()
    model_.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model_


def train_model(model_, t_ds_data, t_ds_label, v_ds_data, v_ds_label, epochs_, types):
    history = model_.fit(
        t_ds_data,
        t_ds_label,
        validation_data=(v_ds_data, v_ds_label),
        epochs=epochs_,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
    )
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()


def test(model_, test_ds_data, test_ds_label, types):
    sum_ = 0.0
    count = 0
    for i in range(len(test_ds_label)):
        predict = model_.predict_on_batch(np.expand_dims(test_ds_data[i], axis=0))
        y_pred = types[np.argmax(predict)]
        y_true = types[test_ds_label[i]]
        sum_ += y_pred == y_true
        count += 1
    test_acc = sum_ / count
    print(f'Test set accuracy: {test_acc:.0%}')


if __name__ == '__main__':
    train_ds_data, train_ds_label, validation_ds_data, validation_ds_label, test_ds_data, test_ds_label, types \
        = prep.get_dataset(DATASET_PATH)
    model = create_model(train_ds_data, types)
    train_model(model, train_ds_data, train_ds_label, validation_ds_data, validation_ds_label, EPOCHS, types)
    test(model, test_ds_data, test_ds_label, types)
