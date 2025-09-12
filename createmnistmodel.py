import numpy as np
import DnnLib
import matplotlib.pyplot as plt

data_train = np.load("/archivos/mnist_train.npz")
data_test = np.load("/archivos/mnist_test.npz")

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

layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(128, 10, DnnLib.ActivationType.SOFTMAX)

optimizer = DnnLib.Adam(learning_rate=0.01)

def train_network(X, y, epochs=10, batch_size=256):
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            h1 = layer1.forward(X_batch)
            output = layer2.forward(h1)

            loss = DnnLib.cross_entropy(output, y_batch)
            epoch_loss += loss
            n_batches += 1

            grad = DnnLib.cross_entropy_gradient(output, y_batch)
            grad = layer2.backward(grad)
            grad = layer1.backward(grad)

            optimizer.update(layer2)
            optimizer.update(layer1)

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if epoch % 2 == 0:
            test_h1 = layer1.forward(test_images)
            test_output = layer2.forward(test_h1)
            test_pred = np.argmax(test_output, axis=1)
            accuracy = np.mean(test_pred == test_labels) * 100
            accuracy_history.append(accuracy)
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return loss_history, accuracy_history


loss_history, accuracy_history = train_network(train_images, train_labels_oh, epochs=10)




plt.figure(figsize=(8, 6))
plt.plot(accuracy_history, loss_history, 'ro-')
plt.title('Pérdida vs Precisión')
plt.xlabel('Precisión (%)')
plt.ylabel('Pérdida')
plt.grid(True)
plt.show()
