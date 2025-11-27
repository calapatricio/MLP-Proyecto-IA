import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


def prepare_dataset(dataset, **kwargs):
    csv = pandas.read_csv(dataset, sep=',')
    csv = csv.dropna()
    data = csv.sample(frac=1., random_state=kwargs['seed'])

    normalise_columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    mms = MinMaxScaler()
    data[normalise_columns] = mms.fit_transform(data[normalise_columns])

    x = (data.drop(columns=['Diabetes_binary'])).values
    y = (data['Diabetes_binary']).values

    valid_p = kwargs['valid_size']
    test_p = kwargs['test_size']

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(valid_p + test_p),
                                                        stratify=y, random_state=kwargs['seed'])
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=test_p / (valid_p + test_p),
                                                        stratify=y_temp, random_state=kwargs['seed'])

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_weights(input_size, hidden_size, hidden_layers):
    network_weights = []
    bias_weights = []

    initial_hidden_weights = np.random.rand(hidden_size, input_size)
    network_weights.append(initial_hidden_weights)

    initial_bias_weights = np.random.rand(hidden_size, 1)
    bias_weights.append(initial_bias_weights)

    for i in range(hidden_layers - 1):
        hidden_weights = np.random.rand(hidden_size, hidden_size)
        network_weights.append(hidden_weights)

        hidden_bias_weights = np.random.rand(hidden_size, 1)
        bias_weights.append(hidden_bias_weights)

    output_weights = np.random.rand(hidden_size)
    network_weights.append(output_weights)

    output_bias_weights = np.array([np.random.rand()])
    bias_weights.append(output_bias_weights)
    return network_weights, bias_weights


def forward_propagation(input_values, network_weights, bias_weights, hidden_size, hidden_layers):
    hidden_layer_outputs = []
    input_bias = 1

    for i in range(hidden_layers):
        actual_hidden_layer_outputs = []
        for j in range(hidden_size):
            neuron_output = sigmoid(
                np.dot(input_values, network_weights[i][j]) + float(np.dot(input_bias, bias_weights[i][j])))
            actual_hidden_layer_outputs.append(neuron_output)

        hidden_layer_outputs.append(np.array(actual_hidden_layer_outputs))
        input_values = hidden_layer_outputs[i]

    output = sigmoid(np.dot(input_values, network_weights[-1]) + float(np.dot(input_bias, bias_weights[-1])))
    return hidden_layer_outputs, output


def backward_propagation(target, learning_rate, input_values, hidden_outputs, output, network_weights, hidden_layers):
    output_loss = target - output
    output_gradient = output_loss * sigmoid_derivative(output)
    gradients = [np.array([output_gradient])]
    weights_delta = []
    bias_delta = []

    for i in reversed(range(hidden_layers)):

        if gradients[0].size == 1:
            actual_weights_delta = learning_rate * hidden_outputs[i].T * gradients[0]
            actual_bias_delta = learning_rate * gradients[0]
        else:
            actual_weights_delta = learning_rate * np.outer(hidden_outputs[i], gradients[0]).T
            actual_bias_delta = (learning_rate * gradients[0]).reshape(-1, 1)

        weights_delta.insert(0, actual_weights_delta)
        bias_delta.insert(0, actual_bias_delta)

        if gradients[0].size == 1:
            output_loss = gradients[0] * network_weights[i + 1].T
        else:
            output_loss = np.dot(gradients[0], network_weights[i + 1].T)

        actual_gradients = []

        for j in range(len(hidden_outputs[i])):
            f = sigmoid_derivative(hidden_outputs[i][j])
            actual_gradients.append(output_loss[j] * f)

        gradients.insert(0, np.array(actual_gradients))

    actual_weights_delta = learning_rate * np.outer(input_values, gradients[0]).T
    actual_bias_delta = (learning_rate * gradients[0]).reshape(-1, 1)

    weights_delta.insert(0, actual_weights_delta)
    bias_delta.insert(0, actual_bias_delta)

    return weights_delta, bias_delta


def update_weights(network_weights, bias_weights, weights_delta, bias_delta):
    for i in range(len(network_weights)):
        network_weights[i] += weights_delta[i]
        bias_weights[i] += bias_delta[i]
    return network_weights, bias_weights


def f1_score(result, prediction, threshold=0.5):
    binary_prediction = []

    for pred in prediction:
        if pred >= threshold:
            binary_prediction.append(1)
        else:
            binary_prediction.append(0)

    tn, fp, fn, tp = confusion_matrix(result, binary_prediction).ravel()

    precision_value = precision(tp, fp)
    recall_value = recall(tp, fn)

    print("Accuracy:" + str((tp+tn)/(tp+tn+fp+fn)))
    print("Precision:" + precision_value)
    print("Recall:" + recall_value)

    if (precision_value + recall_value) == 0:
        return 0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def recall(tp, fn):
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def count_consecutive_same_scores(last_five_scores):
    if len(last_five_scores) == 5:
        return len(set(last_five_scores)) == 1
    return False


def multilayer_perceptron(dataset, learning_rate, hidden_size, hidden_num, **kwargs):
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_dataset(dataset, **kwargs)

    network_weights, bias_weights = initialize_weights(x_train.shape[1], hidden_size, hidden_num)
    f1_train_scores = []
    f1_valid_scores = []
    all_outputs = []

    while not count_consecutive_same_scores(f1_valid_scores[-5:]) or not count_consecutive_same_scores(f1_valid_scores[-5:]):
        for input_values, real_output in zip(x_train, y_train):
            hidden_outputs, output = forward_propagation(input_values, network_weights, bias_weights, hidden_size, hidden_num)
            weights_delta, bias_delta = backward_propagation(real_output, learning_rate, input_values, hidden_outputs, output, network_weights, hidden_num)
            network_weights, bias_weights = update_weights(network_weights, bias_weights, weights_delta, bias_delta)

        for input_values, real_output in zip(x_train, y_train):
            hidden_outputs, output = forward_propagation(input_values, network_weights, bias_weights, hidden_size,
                                                         hidden_num)
            all_outputs.append(output)
        train_error = f1_score(y_train, all_outputs)
        print(train_error)
        f1_train_scores.append(train_error)
        all_outputs.clear()

        for input_values, real_output in zip(x_valid, y_valid):
            hidden_outputs, output = forward_propagation(input_values, network_weights, bias_weights, hidden_size,
                                                         hidden_num)
            all_outputs.append(output)
        valid_error = f1_score(y_valid, all_outputs)
        print(valid_error)
        f1_valid_scores.append(valid_error)
        all_outputs.clear()

    for input_values, real_output in zip(x_test, y_test):
        hidden_outputs, output = forward_propagation(input_values, network_weights, bias_weights, hidden_size,
                                                     hidden_num)
        all_outputs.append(output)
    test_error = f1_score(y_test, all_outputs)

    print(f1_train_scores[-1])
    print(f1_valid_scores[-1])
    print(test_error)

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(f1_train_scores)), f1_train_scores, marker='o', linestyle='-', color='b', label='F1 Train Score')
    plt.plot(range(len(f1_valid_scores)), f1_valid_scores, marker='o', linestyle='-', color='r', label='F1 Valid Score')
    plt.title('Evolución del F1 Score a lo largo del tiempo')
    plt.xlabel('Épocas')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.show()
