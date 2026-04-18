import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(hidden_weights, output_weights, point):
    x = np.array([1, *point])
    hidden_activations = relu(hidden_weights @ x)
    hidden_out = np.array([1, *hidden_activations])
    output = sigmoid(output_weights @ hidden_out)
    return output

def train_step(hidden_weights, output_weights, point, target, learning_rate):
    x = np.array([1, *point])
    hidden_inputs = hidden_weights @ x
    hidden_activations = relu(hidden_inputs)
    hidden_out = np.array([1, *hidden_activations])
    output = sigmoid(output_weights @ hidden_out)

    delta_out = (target - output) * output * (1 - output)

    delta_hidden = (output_weights[:, 1:].T @ delta_out) * relu_derivative(hidden_inputs)

    output_weights += learning_rate * np.outer(delta_out, hidden_out)
    hidden_weights += learning_rate * np.outer(delta_hidden, x)

    return np.sum((target - output) ** 2)

def epoch(hidden_weights, output_weights, X_train, y_train, learning_rate=0.1):
    indices = np.random.permutation(len(X_train))
    total_error = 0
    for i in indices:
        total_error += train_step(hidden_weights, output_weights, X_train[i], y_train[i], learning_rate)
    return total_error / len(X_train)

def evaluate(hidden_weights, output_weights, X_test, y_test):
    correct = 0
    for point, target in zip(X_test, y_test):
        output = predict(hidden_weights, output_weights, point)
        if np.argmax(output) == np.argmax(target):
            correct += 1
    return correct / len(X_test)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target

    targets = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        targets[i, label] = 1

    np.random.seed(42)
    indices = np.random.permutation(150)
    test_idx, train_idx = indices[:30], indices[30:]

    X_train, y_train = X[train_idx], targets[train_idx]
    X_test, y_test = X[test_idx], targets[test_idx]

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    n_hidden = 6
    n_inputs = 4
    hidden_weights = np.random.uniform(-1, 1, (n_hidden, n_inputs + 1))
    output_weights = np.random.uniform(-1, 1, (3, n_hidden + 1))

    errors = []
    for e in range(500):
        err = epoch(hidden_weights, output_weights, X_train, y_train)
        errors.append(err)
        if e % 50 == 0:
            print(f"epoch {e:>4}  avg error: {err:.4f}")

    acc = evaluate(hidden_weights, output_weights, X_test, y_test)
    print(f"\nTest accuracy: {acc:.2%}")

    plt.plot(errors)
    plt.title("Training Error per Epoch (ReLU)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Squared Error")
    plt.tight_layout()
    plt.savefig("plots/iris_relu_training_error.png", dpi=150)
    plt.show()
