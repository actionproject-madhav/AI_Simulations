import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_network(activation_fn, activation_derivative, n_epochs=500):
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
    for e in range(n_epochs):
        indices = np.random.permutation(len(X_train))
        total_error = 0
        for i in indices:
            point = X_train[i]
            target = y_train[i]

            x = np.array([1, *point])
            hidden_inputs = hidden_weights @ x
            hidden_activations = activation_fn(hidden_inputs)
            hidden_out = np.array([1, *hidden_activations])
            output = sigmoid(output_weights @ hidden_out)

            delta_out = (target - output) * output * (1 - output)
            delta_hidden = (output_weights[:, 1:].T @ delta_out) * activation_derivative(hidden_inputs)

            output_weights += 0.1 * np.outer(delta_out, hidden_out)
            hidden_weights += 0.1 * np.outer(delta_hidden, x)

            total_error += np.sum((target - output) ** 2)

        errors.append(total_error / len(X_train))

    return errors

# train both versions
sigmoid_errors = train_network(sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)))
relu_errors = train_network(relu, relu_derivative)

# plot comparison
plt.figure(figsize=(10, 5))
plt.plot(sigmoid_errors, label='Sigmoid', alpha=0.8)
plt.plot(relu_errors, label='ReLU', alpha=0.8)
plt.title("Sigmoid vs ReLU Training Error")
plt.xlabel("Epoch")
plt.ylabel("Avg Squared Error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/activation_comparison.png", dpi=150)
plt.show()

print(f"Final sigmoid error: {sigmoid_errors[-1]:.4f}")
print(f"Final ReLU error: {relu_errors[-1]:.4f}")
