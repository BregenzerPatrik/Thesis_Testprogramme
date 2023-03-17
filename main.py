import numpy as np
import matplotlib.pyplot as plt
import pickle


class Neuron:
    def __init__(self, input_size, weights=None, bias=None, activation='relu', name=None):
        self.name = name
        if weights is None:
            self.weights = np.random.randn(input_size)
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.random.randn()
        else:
            self.bias = bias

        if activation == 'sigmoid':
            self.activation = self._sigmoid
        elif activation == 'relu':
            self.activation = self._relu
        else:
            raise ValueError(f"Activation function '{activation}' is not supported")

    def forward(self, inputs):
        # Calculate weighted sum of inputs and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function
        output = self.activation(weighted_sum)
        return output

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        return np.maximum(0, x)

    def visualize_weights(self):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the weights
        ax1.bar(range(len(self.weights)), self.weights)
        ax1.set_xlabel('Input Connection')
        ax1.set_ylabel('Weight')
        ax1.set_title(f'Neuron Weights of Neuron {self.name}')

        # Plot the bias
        ax2.bar([0], [self.bias])
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Bias'])
        ax2.set_ylabel('Value')
        ax2.set_title(f'Neuron Bias of Neuron {self.name}')

        # Show the plot
        plt.show()

    def save_weights_bias(self):
        with open(f'{self.name}.pkl', 'wb') as f:
            pickle.dump((self.weights, self.bias), f)

    def load_weights_bias(self):
        with open(f'{self.name}.pkl', 'rb') as f:
            self.weights, self.bias = pickle.load(f)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        all_outputs = []
        # Propagate inputs through all layers
        for layer in self.layers:
            next_input = []
            for neuron in layer:
                next_input.append(neuron.forward(inputs))
            all_outputs.append(np.array(next_input))
            inputs = np.array(next_input)
            # Return output of last layer and all layer outputs
        return inputs, all_outputs

    def save_weights(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.save_weights_bias()

    def load_weights(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.load_weights_bias()

    def plot_network(self):
        neurons = []
        for layer in self.layers:
            for neuron in layer:
                neurons.append(neuron)
        num_neurons = len(neurons)
        weights_data = [neuron.weights for neuron in neurons]
        biases_data = [neuron.bias for neuron in neurons]
        x_labels = [neuron.name for neuron in neurons]

        fig, ax = plt.subplots()
        bar_width = 0.4
        opacity = 0.8

        index = np.arange(num_neurons)
        i = 0
        for weights in weights_data:
            j = 0
            for weight in weights:
                ax.bar(index[i] + j * bar_width / len(weights), weight, bar_width / len(weights), alpha=opacity,
                       color='b')
                j = j + 1 % len(colors)
            ax.bar(index[i] + bar_width, biases_data[i], bar_width, alpha=opacity, color='r')
            i = i + 1

        ax.set_xlabel('Neuron')
        ax.set_ylabel('Value')
        ax.set_title('Network Weights and Biases')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(x_labels)
        ax.legend()

        plt.tight_layout()
        plt.show()


# set custom weights
weights_n1 = np.array([0.7, 1.2, 1.5])
weights_n2 = np.array([0.4, 0.3, 0.2])
weights_n3 = np.array([1.2, 0.5])
weights_n4 = np.array([0.3, 0.9])
weights_n5 = np.array([0.4, 0.9])
weights_n6 = np.array([0.3, 0.2])
# set custom bias
bias_n1 = 0.3
bias_n2 = 0.7
bias_n3 = 2
bias_n4 = 3
bias_n5 = 0.1
bias_n6 = 0.2

# Set input values
inputs = np.array([3, -5, 10])
# Set neurons
n1 = Neuron(3, weights_n1, bias_n1)
n2 = Neuron(3, weights_n2, bias_n2)
n3 = Neuron(2, weights_n3, bias_n3)
n4 = Neuron(2, weights_n4, bias_n4)
n5 = Neuron(2, weights_n5, bias_n5, activation='sigmoid')
n6 = Neuron(2, weights_n6, bias_n6, activation='sigmoid')

# Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
network = NeuralNetwork([
    [n1, n2],
    [n3, n4],
    [n5, n6]
])

# Compute the output of the network for a given input
output, all_outputs = network.forward([3, -5, 10])
print("Output: ", output)
print("All Layers: ", all_outputs)

# bigger network
p1 = Neuron(6, name='p1')
p2 = Neuron(6, name='p2')
p3 = Neuron(6, name='p3')
p4 = Neuron(6, name='p4')
q1 = Neuron(4, name='q1')
q2 = Neuron(4, name='q2')
q3 = Neuron(4, name='q3')
q4 = Neuron(4, name='q4')
r1 = Neuron(4, name='r1', activation='sigmoid')
r2 = Neuron(4, name='r2', activation='sigmoid')
r3 = Neuron(4, name='r3', activation='sigmoid')

bigger_network = NeuralNetwork([
    [p1, p2, p3, p4],
    [q1, q2, q3, q4],
    [r1, r2, r3]
])
# bigger_network.save_weights()
bigger_network.load_weights()
bigger_network.plot_network()

output, all_outputs = bigger_network.forward([4, 7, 15, 2, -7, 3])
print("Output: ", output)
print("All Layers: ", all_outputs)
