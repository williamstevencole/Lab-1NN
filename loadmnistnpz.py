import numpy as np
import matplotlib.pyplot as plt
import os

data = np.load(os.path.join(os.path.dirname(__file__), 'mnist_train.npz'))

images = data['images']
labels = data['labels']

images = images.astype(np.float32) / 255.0

print(f"Shape de las im�genes: {images.shape}")
print(f"Shape de las etiquetas: {labels.shape}")

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

random_indices = np.random.choice(len(images), 3, replace=False)

for i, idx in enumerate(random_indices):
  axes[i].imshow(images[idx], cmap='gray')
  axes[i].set_title(f'Dígito: {labels[idx]}')
  axes[i].axis('off')

plt.tight_layout()
plt.show()
