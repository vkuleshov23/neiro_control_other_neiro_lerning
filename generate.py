import numpy as np
import matplotlib.pyplot as plt
import csv


FILES_COUNT = 3
EPOCHS = 100

LOSS_SCALE = 0.35
LOSS_MIN = 1.2

ACC_MIN = 0.1
ACC_MAX = 0.4

TRAINED_STUDY_LOC = 0.85
TRAINED_STUDY_SACLE = 0.04

RETRAINED_STUDY_LOC = 0.5
RETRAINED_STUDY_SACLE = 0.04
RETRAINED_STAGNATION_LOC = 0.27
RETRAINED_STAGNATION_SACLE = 0.04

diaps_for_study = [[0.97, 0.98], [0.975, 0.98], [0.978, 0.982]]
diaps_for_stagnation_loss = [[-0.002, 0.0022], [-0.001, 0.002], [-0.002, 0.002]]
diaps_for_stagnation_acc = [[0.0002, -0.0022], [0.002, -0.002], [0.0002, -0.002]]
diaps_for_retrain = [[-0.015, 0.015], [-0.005, 0.005], [-0.01, 0.01]]

generate
def generate_random_loss(size: int = 1):
    return np.random.exponential(size=size, scale=LOSS_SCALE) + LOSS_MIN


def generate_random_accuracy(size: int = 1):
    return np.random.uniform(ACC_MIN, ACC_MAX, size)


def split_epoch_trained(epochs: int):
    study = np.random.normal(loc=TRAINED_STUDY_LOC, scale=TRAINED_STUDY_SACLE)
    study_epochs = int(epochs * study)
    epochs = epochs - study_epochs
    stagnation = epochs
    return study_epochs, stagnation, 0


def split_epoch_understudied(epochs: int):
    return epochs, 0, 0


def split_epoch_retrained(epochs: int):
    study = np.random.normal(loc=RETRAINED_STUDY_LOC, scale=RETRAINED_STUDY_SACLE)
    study_epochs = int(epochs * study)
    epochs = epochs - study_epochs
    stagnation = np.random.normal(loc=RETRAINED_STAGNATION_LOC, scale=RETRAINED_STAGNATION_SACLE)
    stagnation_epochs = int(epochs * stagnation)
    epochs = epochs - stagnation_epochs
    return study_epochs, stagnation_epochs, epochs


def decrease_loss(loss, epochs):
    pos = np.random.randint(0, len(diaps_for_study))
    losses = []
    for i in range(epochs):
        losses.append(loss)
        loss = loss * np.random.uniform(diaps_for_study[pos][0], diaps_for_study[pos][1])
    return losses[len(losses)-1], losses


def increase_acc(accuracy, epochs):
    accuracies = []
    increase_range = 1 - accuracy
    pos = np.random.randint(0, len(diaps_for_study))
    for i in range(epochs):
        accuracies.append(accuracy)
        accuracy = 1 - increase_range
        increase_range = increase_range * np.random.uniform(diaps_for_study[pos][0], diaps_for_study[pos][1])
    return accuracies[len(accuracies)-1], accuracies


def stagnate_params_loss(param, epochs):
    params = []
    pos = np.random.randint(0, len(diaps_for_stagnation_loss))
    for i in range(epochs):
        param = param - np.random.uniform(diaps_for_stagnation_loss[pos][0], diaps_for_stagnation_loss[pos][1])
        params.append(param)
    return param, params


def stagnate_params_acc(param, epochs):
    params = []
    pos = np.random.randint(0, len(diaps_for_stagnation_acc))
    for i in range(epochs):
        param = param - np.random.uniform(diaps_for_stagnation_acc[pos][0], diaps_for_stagnation_acc[pos][1])
        params.append(param)
    return param, params


def retrain_params(param, epochs):
    params = []
    pos = np.random.randint(0, len(diaps_for_retrain))
    for i in range(epochs):
        param = param + np.random.uniform(diaps_for_retrain[pos][0], diaps_for_retrain[pos][1])
        params.append(param)
    return param, params


def train_process(epochs, loss, val_loss, accuracy, val_accuracy):
    loss, losses = decrease_loss(loss, epochs)
    val_loss, val_losses = decrease_loss(val_loss, epochs)
    accuracy, accuracies = increase_acc(accuracy, epochs)
    val_accuracy, val_accuracies = increase_acc(val_accuracy, epochs)
    return losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy


def stagnation_process(epochs, loss, val_loss, accuracy, val_accuracy):
    loss, losses = stagnate_params_loss(loss, epochs)
    val_loss, val_losses = stagnate_params_loss(val_loss, epochs)
    accuracy, accuracies = stagnate_params_acc(accuracy, epochs)
    val_accuracy, val_accuracies = stagnate_params_acc(val_accuracy, epochs)

    return losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy


def retrain_process(epochs, loss, val_loss, accuracy, val_accuracy):
    loss, losses = retrain_params(loss, epochs)
    val_loss, val_losses = retrain_params(val_loss, epochs)
    accuracy, accuracies = retrain_params(accuracy, epochs)
    val_accuracy, val_accuracies = retrain_params(val_accuracy, epochs)

    return losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy


def generate_metrics():
    return generate_random_loss(1), generate_random_loss(1), generate_random_accuracy(1), generate_random_accuracy(1)


def generate_trained(epochs: int):
    loss, val_loss, accuracy, val_accuracy = generate_metrics()
    study, stagnation, over = split_epoch_trained(epochs)
    losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy \
        = train_process(study, loss, val_loss, accuracy, val_accuracy)
    losses_stag, val_losses_stag, accuracies_stag, val_accuracies_stag, loss, val_loss, accuracy, val_accuracy \
        = stagnation_process(stagnation, loss, val_loss, accuracy, val_accuracy)
    return losses + losses_stag, val_losses + val_losses_stag, accuracies + accuracies_stag, val_accuracies + val_accuracies_stag


def generate_understudied(epochs: int):
    loss, val_loss, accuracy, val_accuracy = generate_metrics()
    study, stagnation, over = split_epoch_understudied(epochs)
    losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy \
        = train_process(study, loss, val_loss, accuracy, val_accuracy)
    return losses, val_losses, accuracies, val_accuracies


def generate_retrained(epochs: int):
    loss, val_loss, accuracy, val_accuracy = generate_metrics()
    study, stagnation, over = split_epoch_retrained(epochs)
    losses, val_losses, accuracies, val_accuracies, loss, val_loss, accuracy, val_accuracy \
        = train_process(study, loss, val_loss, accuracy, val_accuracy)
    losses_stag, val_losses_stag, accuracies_stag, val_accuracies_stag, loss, val_loss, accuracy, val_accuracy \
        = stagnation_process(stagnation, loss, val_loss, accuracy, val_accuracy)
    losses = losses + losses_stag
    val_losses = val_losses + val_losses_stag
    accuracies = accuracies + accuracies_stag
    val_accuracies = val_accuracies + val_accuracies_stag
    losses_retain, val_losses_retain, accuracies_retain, val_accuracies_retain, loss, val_loss, accuracy, val_accuracy \
        = retrain_process(over, loss, val_loss, accuracy, val_accuracy)
    return losses + losses_retain, val_losses + val_losses_retain, accuracies + accuracies_retain, val_accuracies + val_accuracies_retain


def createCSV(loss, val_loss, acc, val_acc, epoch: int, file_name):
    with open(file_name, 'w') as file:
        wr = csv.writer(file)
        head = ["loss", "val_loss", "acc", "val_acc"]
        wr.writerow(head)
        for i in range(epoch):
            line = [float(loss[i]), float(val_loss[i]), float(acc[i]), float(val_acc[i])]
            wr.writerow(line)


if __name__ == '__main__':
    for i in range(FILES_COUNT):
        # loss, val_loss, accuracy, val_accuracy = generate_understudied(EPOCHS)
        # createCSV(loss, val_loss, accuracy, val_accuracy, EPOCHS, "data/understudied/file" + str(i) + ".csv")
        # loss, val_loss, accuracy, val_accuracy = generate_trained(EPOCHS)
        # createCSV(loss, val_loss, accuracy, val_accuracy, EPOCHS, "data/trained/file" + str(i) + ".csv")
        loss, val_loss, accuracy, val_accuracy = generate_retrained(EPOCHS)
        # createCSV(loss, val_loss, accuracy, val_accuracy, EPOCHS, "data/retrained/file" + str(i) + ".csv")
        plt.plot(loss)
        plt.plot(val_loss)
        plt.plot(accuracy)
        plt.plot(val_accuracy)
        plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
        plt.show()
