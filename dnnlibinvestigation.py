import numpy as np
import DnnLib

hidden_layer = DnnLib.DenseLayer(784, 64, DnnLib.ActivationType.RELU)
output_layer = DnnLib.DenseLayer(64, 10, DnnLib.ActivationType.SOFTMAX)

x = np.array([0.5, -0.2, 0.1, 2.0, -1.0])

print("Testing activation functions:")
print("ReLU:", DnnLib.relu(x))
print("Sigmoid:", DnnLib.sigmoid(x))
print("Tanh:", DnnLib.tanh(x))

print("\nGeneric activation:")
print("ReLU generic:", DnnLib.apply_activation(x, DnnLib.ActivationType.RELU))
print("Sigmoid generic:", DnnLib.apply_activation(x, DnnLib.ActivationType.SIGMOID))

print("\nDerivatives:")
print("ReLU derivative:", DnnLib.relu_derivative(x))
print("Sigmoid derivative:", DnnLib.sigmoid_derivative(x))

test_layer = DnnLib.DenseLayer(3, 2, DnnLib.ActivationType.RELU)
test_layer.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
test_layer.bias = np.array([0.01, -0.02])

test_input = np.array([[0.5, -0.2, 0.1]])
print("\nLayer testing:")
print("With activation:", test_layer.forward(test_input))
print("Linear (no activation):", test_layer.forward_linear(test_input))

