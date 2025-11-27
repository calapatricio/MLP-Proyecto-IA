import numpy as np
from src.mlp_algorithm import multilayer_perceptron


def train_and_test(dataset, seed=0, learning_rate=.0, hidden_size=0, hidden_num=0):
    """
    Returns the test metric after training with a partition of the dataset
    :param dataset: pandas dataset to train the network with
    :param seed: (int) used to initialize the random number generator
    :param learning_rate: (float) learning rate of the MLP
    :param hidden_size: (int) number of neurons in each hidden layer
    :param hidden_num: (int) number of hidden layers in the network
    :return: the testing error, a float number
    """

    np.random.seed(seed)
    valid_size = 0.05
    test_size = 0.05

    return multilayer_perceptron(dataset, learning_rate, hidden_size, hidden_num, seed=seed, valid_size=valid_size, test_size=test_size)

train_and_test('diabetes_binary_health_indicators_BRFSS2015.csv', 25, 1, 10, 1)
