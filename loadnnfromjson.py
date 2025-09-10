import json
import numpy as np
import DnnLib

def load_neural_network(json_file, layers):
    """Load weights and biases from JSON into existing DnnLib layers"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    for i, layer_data in enumerate(data['layers']):
        weights_matrix = np.array(layer_data['W'])
        bias_vector = np.array(layer_data['b'])
        layers[i].weights = weights_matrix.T
        layers[i].bias = bias_vector

    return data






if __name__ == "__main__":
    # Define the neural network architecture
    layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
    layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)
    layers = [layer1, layer2]
    json_data = load_neural_network('mnist_mlp_pretty.json', layers)

    # Load and preprocess test data
    test_data = np.load('mnist_test.npz')
    test_images = test_data['images']
    test_labels = test_data['labels']
    test_images = test_images.astype(np.float32) / 255.0

    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    x = test_images_flat

    # Forward pass through the network
    for i, layer in enumerate(layers):
        x = layer.forward(x)

    # Get predictions and calculate accuracy
    predictions = np.argmax(x, axis=1)
    correct = np.sum(predictions == test_labels)
    total = len(test_labels)
    accuracy = correct / total

    print(f"\nResults:")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


    print(f"\nFirst 50 predictions vs actual:")
    for i in range(50):
        print(f"Predicted: {predictions[i]}, Actual: {test_labels[i]}, {'✓' if predictions[i] == test_labels[i] else '✗'}")
