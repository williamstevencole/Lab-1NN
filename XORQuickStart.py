import DnnLib
import numpy as np

# Create sample data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[0], [1], [1], [0]], dtype=np.float64)

# Create a neural network: 2 -> 4 -> 1
layer1 = DnnLib.DenseLayer(2, 4, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(4, 1, DnnLib.ActivationType.SIGMOID)

# Create optimizer
optimizer = DnnLib.Adam(learning_rate=0.01)

for epoch in range(100):
  # Forward pass
  h1 = layer1.forward(X)
  output = layer2.forward(h1)
  # Compute loss
  loss = DnnLib.mse(output, y)
  # Backward pass
  loss_grad = DnnLib.mse_gradient(output, y)
  grad2 = layer2.backward(loss_grad)
  grad1 = layer1.backward(grad2)
  # Update parameters
  optimizer.update(layer2)
  optimizer.update(layer1)
  if epoch % 20 == 0:
    print(f"Epoch {epoch}, Loss: {loss:.6f}")
