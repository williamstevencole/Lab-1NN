import numpy as np
import DnnLib
import matplotlib.pyplot as plt

# Carga (.npz) y preprocesado simple [0,1]
data_train = np.load("/archivos/mnist_train.npz")
data_test  = np.load("/archivos/mnist_test.npz")

train_images = data_train['images'].reshape(-1, 784).astype(np.float64)/255.0
train_labels = data_train['labels'].astype(np.int64)

test_images  = data_test['images'].reshape(-1, 784).astype(np.float64)/255.0
test_labels  = data_test['labels'].astype(np.int64)

def one_hot(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes), dtype=np.float64)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

# Split 80/20 para validación
rng = np.random.default_rng(0)
N = len(train_images)
idx = rng.permutation(N)
val_n = int(0.20 * N)
val_images, val_labels = train_images[idx[:val_n]], train_labels[idx[:val_n]]
train_images, train_labels = train_images[idx[val_n:]], train_labels[idx[val_n:]]

train_labels_oh = one_hot(train_labels)
val_labels_oh   = one_hot(val_labels)

layer1 = DnnLib.DenseLayer(784, 128, DnnLib.ActivationType.RELU)
layer2 = DnnLib.DenseLayer(128, 10,  DnnLib.ActivationType.SOFTMAX)
optimizer = DnnLib.Adam(learning_rate=1e-4)

def evaluate_accuracy(X, y_int):
    h1 = layer1.forward(X)
    out = layer2.forward(h1)
    pred = np.argmax(out, axis=1)
    return (pred == y_int).mean() * 100.0

def train_network(X, y, epochs=25, batch_size=64):
    loss_history, val_acc_history = [], []
    rng_local = np.random.default_rng(1)

    for epoch in range(1, epochs+1):
        perm = rng_local.permutation(len(X))
        X, y = X[perm], y[perm]

        epoch_loss, n_batches = 0.0, 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            h1 = layer1.forward(xb)
            out = layer2.forward(h1)
            loss = DnnLib.cross_entropy(out, yb)
            grad = DnnLib.cross_entropy_gradient(out, yb)
            grad = layer2.backward(grad)
            grad = layer1.backward(grad)
            optimizer.update(layer2); optimizer.update(layer1)
            epoch_loss += loss; n_batches += 1

        avg_loss = epoch_loss / n_batches
        val_acc = evaluate_accuracy(val_images, val_labels)
        loss_history.append(avg_loss); val_acc_history.append(val_acc)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return loss_history, val_acc_history

loss_history, accuracy_history = train_network(train_images, train_labels_oh, epochs=25, batch_size=64)

test_acc = evaluate_accuracy(test_images, test_labels)
print(f"\nTest Accuracy: {test_acc:.2f}%")

ep = range(1, len(loss_history)+1)
plt.figure(figsize=(8,5))
plt.plot(ep, loss_history, 'o-', label='Loss')
plt.plot(ep, accuracy_history, 's-', label='Val Acc (%)')
plt.xlabel('Época'); plt.grid(True); plt.legend(); plt.title('MNIST (2 capas, Adam 1e-4)')
plt.show()
