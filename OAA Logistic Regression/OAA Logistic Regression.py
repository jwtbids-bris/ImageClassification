import torch
import numpy as np
import time
from torchvision import datasets, transforms

# LOAD DATA
print("Loading MNIST...")

train_data = datasets.MNIST("./data", train=True,  download=True)
test_data  = datasets.MNIST("./data", train=False, download=True)

# Convert to tensors, normalise pixels to [0, 1]
X_train = train_data.data.reshape(-1, 784).float() / 255.0
y_train = train_data.targets
X_test  = test_data.data.reshape(-1, 784).float() / 255.0
y_test  = test_data.targets

print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

# HYPERPARAMETERS
lr         = 0.5
epochs     = 50
batch_size = 128
D          = 784
sizes      = [1000, 5000, 10000, 20000, 40000, 60000]

# BENCHMARK ACROSS SIZES
results = []

for N in sizes:

    X = X_train[:N]
    y = y_train[:N]

    all_weights = []
    all_biases  = []

    start_time = time.time()

    # Train 10 binary classifiers
    for digit in range(10):

        y_binary = (y == digit).float()

        w = torch.zeros(D)
        b = torch.tensor(0.0)

        for epoch in range(epochs):

            # Shuffle data each epoch
            perm = torch.randperm(N)

            # Loop over mini-batches
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = perm[start:end]
                B   = end - start  # actual batch size

                X_batch = X[idx]
                y_batch = y_binary[idx]

                #  Forward pass
                z     = X_batch @ w + b
                y_hat = torch.sigmoid(z)

                # Binary cross-entropy loss
                eps  = 1e-7
                loss = -torch.mean(
                    y_batch * torch.log(y_hat + eps) +
                    (1 - y_batch) * torch.log(1 - y_hat + eps)
                )

                # Gradients
                error = y_hat - y_batch
                dw    = (1/B) * (X_batch.T @ error)
                db    = (1/B) * torch.sum(error)

                # Update weights 
                w = w - lr * dw
                b = b - lr * db

        all_weights.append(w)
        all_biases.append(b)

    train_time = time.time() - start_time

    # Test
    W = torch.stack(all_weights, dim=1)
    Bi = torch.stack(all_biases)
    scores = X_test @ W + Bi
    predictions = torch.argmax(scores, dim=1)
    correct  = (predictions == y_test).sum().item()
    accuracy = correct / len(y_test)

    results.append((N, train_time, accuracy))
    print(f"N = {N:>6,d}  |  Time: {train_time:>7.2f}s  |  Accuracy: {accuracy:.2f}")

# PRINT RESULTS
print("\n" + "=" * 50)
print(f"{'Size (N)':>10s}  {'Training Time (s)':>18s}  {'Accuracy':>10s}")
print("-" * 50)
for N, t, a in results:
    print(f"{N:>10,d}  {t:>18.2f}  {a:>10.2f}")
print("=" * 50)