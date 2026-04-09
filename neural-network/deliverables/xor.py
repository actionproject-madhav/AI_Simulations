import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(hidden_weights, output_weights, point):
    x = np.array([1, *point])
    hidden_activations = sigmoid(hidden_weights @ x)
    hidden_out = np.array([1, *hidden_activations])
    output = sigmoid(output_weights @ hidden_out)
    return output

def train(points, labels, hidden_weights, output_weights, learning_rate=0.1, epochs=10000):
    for _ in range(epochs):
        for point, label in zip(points, labels):
            # forward pass
            x = np.array([1, *point])
            hidden_activations = sigmoid(hidden_weights @ x)
            hidden_out = np.array([1, *hidden_activations])
            output = sigmoid(output_weights @ hidden_out)

            # output delta
            delta_out = (label - output) * output * (1 - output)

            # hidden delta — chain through output weights (skip bias weight)
            delta_hidden = output_weights[1:] * delta_out * hidden_activations * (1 - hidden_activations)

            # update weights
            output_weights += learning_rate * delta_out * hidden_out
            hidden_weights += learning_rate * np.outer(delta_hidden, x)

    return hidden_weights, output_weights

if __name__ == "__main__":
    np.random.seed(0)
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])

    hidden_weights = np.random.uniform(-1, 1, (4, 3))  # 4 hidden nodes, bias + 2 inputs
    output_weights = np.random.uniform(-1, 1, 5)       # bias + 4 hidden nodes

    hidden_weights, output_weights = train(points, labels, hidden_weights, output_weights)

    for point, label in zip(points, labels):
        out = predict(hidden_weights, output_weights, point)
        print(f"{point} -> {out:.4f}  (expected {label})")

    