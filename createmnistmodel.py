import numpy as np
import DnnLib
import json

data_train = np.load("mnist_train.npz")
data_test = np.load("mnist_test.npz")

train_images = data_train['images'].reshape(-1, 784) / 255.0
train_labels = data_train['labels']

test_images = data_test['images'].reshape(-1, 784) / 255.0
test_labels = data_test['labels']

def one_hot(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded

train_labels_oh = one_hot(train_labels)
test_labels_oh = one_hot(test_labels)

layers = []

hidden_layer = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
hidden_layer.weights = np.random.randn(784, 128) * np.sqrt(2.0 / 784)
hidden_layer.bias = np.zeros(128)
layers.append(hidden_layer)

output_layer = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
output_layer.weights = np.random.randn(128, 10) * 0.01
output_layer.bias = np.zeros(10)
layers.append(output_layer)

# for layer in layers:
#     layer.Adam(0.001)

def adam_optimizer(weights, bias, grad_w, grad_b, m_w, v_w, m_b, v_b, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
    m_w = beta1 * m_w + (1 - beta1) * grad_w
    v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
    m_b = beta1 * m_b + (1 - beta1) * grad_b
    v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

    m_w_corrected = m_w / (1 - beta1 ** t)
    v_w_corrected = v_w / (1 - beta2 ** t)
    m_b_corrected = m_b / (1 - beta1 ** t)
    v_b_corrected = v_b / (1 - beta2 ** t)

    weights -= lr * m_w_corrected / (np.sqrt(v_w_corrected) + epsilon)
    bias -= lr * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

    return weights, bias, m_w, v_w, m_b, v_b

adam_states = []
for layer in layers:
    adam_states.append({
        'm_w': np.zeros_like(layer.weights), # Initialize first moment vector for weights
        'v_w': np.zeros_like(layer.weights), # Initialize second moment vector for weights
        'm_b': np.zeros_like(layer.bias),    # Initialize first moment vector for bias
        'v_b': np.zeros_like(layer.bias),    # Initialize second moment vector for bias
        't': 0 # Time step
    })

def forward(x):
    activations = [x]
    for layer in layers:
        z = np.dot(activations[-1], layer.weights) + layer.bias
        a = DnnLib.apply_activation(z, layer.activation_type)
        activations.append(a)
    return activations

def backward(activations, y_true):
    delta = activations[-1] - y_true

    for i in reversed(range(len(layers))):
        layer = layers[i]

        grad_w = np.dot(activations[i].T, delta) / y_true.shape[0]
        grad_b = np.mean(delta, axis=0)

        # layer.update_weights(grad_w, grad_b)

        state = adam_states[i]
        state['t'] += 1

        layer.weights, layer.bias, state['m_w'], state['v_w'], state['m_b'], state['v_b'] = adam_optimizer(
            layer.weights, layer.bias, grad_w, grad_b,
            state['m_w'], state['v_w'], state['m_b'], state['v_b'],
            t=state['t']
        )

        if i > 0:
            delta = np.dot(delta, layer.weights.T)
            z = np.dot(activations[i-1], layers[i-1].weights) + layers[i-1].bias
            delta *= DnnLib.apply_activation_derivative(z, layers[i-1].activation_type)

epochs = 10
batch_size = 32

# Training
for epoch in range(epochs):
    indices = np.random.permutation(len(train_images))
    train_images_shuffled = train_images[indices]
    train_labels_shuffled = train_labels_oh[indices]

    epoch_loss = 0
    for i in range(0, len(train_images), batch_size):
        batch_x = train_images_shuffled[i:i+batch_size]
        batch_y = train_labels_shuffled[i:i+batch_size]

        activations = forward(batch_x)
        loss = -np.mean(batch_y * np.log(np.clip(activations[-1], 1e-15, 1-1e-15)))
        epoch_loss += loss

        backward(activations, batch_y)

    test_activations = forward(test_images)
    test_pred = np.argmax(test_activations[-1], axis=1)
    accuracy = np.mean(test_pred == test_labels) * 100

    print(f"Epoch {epoch+1} | Loss: {epoch_loss/(len(train_images)//batch_size):.4f} | Accuracy: {accuracy:.2f}%")
