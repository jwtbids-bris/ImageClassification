import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10

def rgb2gray(images):
    """
    Convert RGB images to grayscale.
    formula: Y = 0.299R + 0.587G + 0.114B
    Input shape: (N, 32, 32, 3)
    Output shape: (N, 1024)  (flattened)
    """
    # 0.299 * R + 0.587 * G + 0.114 * B
    gray_images = np.dot(images[...,:3], [0.299, 0.587, 0.114])
    # Flatten: (N, 32, 32) -> (N, 1024)
    return gray_images.reshape(images.shape[0], -1)

def main():
    print("Loading CIFAR-10 data...")
    (X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = cifar10.load_data()
    
    # Flatten y to 1D array
    y_train_orig = y_train_orig.ravel()
    y_test_orig = y_test_orig.ravel()

    print(f"Original Training Shape: {X_train_orig.shape}")
    print(f"Original Test Shape: {X_test_orig.shape}")

    # --- Preprocessing ---
    print("\n preprocessing: Flattening RGB images (no grayscale)...")
    # Flatten: (N, 32, 32, 3) -> (N, 3072)
    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1)
    X_test = X_test_orig.reshape(X_test_orig.shape[0], -1)
    
    y_train = y_train_orig
    y_test = y_test_orig

    # Optimized Setting: Full Dataset
    TRAIN_SIZE = 50000
    print(f"Using full training data: {TRAIN_SIZE} samples")
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Benchmarking ---
    print("\n--- Starting Benchmark ---")

    # --- Benchmarking ---
    print("\n--- Starting Benchmark ---")

    # 1. Linear SVM
    print("\nTraining LinearSVC (Full Data, RGB)...")
    start = time.time()
    # dual=False is preferred when n_samples > n_features
    global_tol = 1e-3
    linear_clf = LinearSVC(dual=False, C=1.0, max_iter=1000, tol=global_tol)
    linear_clf.fit(X_train, y_train)
    linear_time = time.time() - start
    linear_acc = linear_clf.score(X_test, y_test)
    print(f"LinearSVC Time: {linear_time:.2f}s")
    print(f"LinearSVC Accuracy: {linear_acc:.4f}")

    # 2. RBF SVM
    print("\nTraining RBF SVM (Full Data, RGB)...")
    print("WARNING: This may take a very long time (30+ minutes) due to O(N^2*D) complexity.")
    start = time.time()
    # Using cache_size=1000 (MB) to speed up if memory allows
    rbf_clf = SVC(kernel='rbf', C=1.0, gamma='scale', tol=global_tol, cache_size=1000)
    rbf_clf.fit(X_train, y_train)
    rbf_time = time.time() - start
    rbf_acc = rbf_clf.score(X_test, y_test)
    print(f"RBF SVM Time: {rbf_time:.2f}s")
    print(f"RBF SVM Accuracy: {rbf_acc:.4f}")

    # 3. MLP Classifier (Optimized)
    print("\nTraining MLP (Optimized Architecture)...")
    start = time.time()
    # Deeper architecture: 1024 -> 512 -> 256 -> Output
    mlp_clf = MLPClassifier(hidden_layer_sizes=(1024, 512, 256), activation='relu', 
                            solver='adam', max_iter=500, random_state=42, early_stopping=True)
    mlp_clf.fit(X_train, y_train)
    mlp_time = time.time() - start
    mlp_acc = mlp_clf.score(X_test, y_test)
    print(f"MLP Time: {mlp_time:.2f}s")            
    print(f"MLP Accuracy: {mlp_acc:.4f}")

    # --- Summary ---
    print("\n" + "="*60)
    print("CIFAR-10 Final Benchmark (50k Samples, RGB)")
    print("="*60)
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Time (s)':<10}")
    print("-" * 60)
    print(f"{'LinearSVC':<25} | {linear_acc:.4f}     | {linear_time:.2f}")
    print(f"{'RBF SVM':<25} | {rbf_acc:.4f}     | {rbf_time:.2f}")
    print(f"{'MLP (1024,512,256)':<25} | {mlp_acc:.4f}     | {mlp_time:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
