import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

def load_data():
    """Load and preprocess MNIST data."""
    print("Loading MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Split train/test first to avoid data leakage
    # We reserve 10k for testing to check accuracy
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )
    return X_train_full, X_test, y_train_full, y_test

def run_benchmark(model_factory, X_train, y_train, X_test, y_test, num_trials=3):
    """
    Run benchmark for a specific model configuration.
    Returns average training time and accuracy.
    """
    times = []
    accuracies = []
    
    for _ in range(num_trials):
        model = model_factory()
        
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        
        times.append(end - start)
        accuracies.append(model.score(X_test, y_test))
    
    return np.mean(times), np.mean(accuracies)

def main():
    X_train_full, X_test_raw, y_train_full, y_test = load_data()
    
    # scale data in the training set
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    
    # scale data in the testing set
    X_test = scaler.transform(X_test_raw)

    # Benchmark Configuration of different sizes of training set
    subset_sizes = [1000, 5000, 10000, 20000, 40000, 60000] 
    n_trials = 3
    
    results = {
        'linear': {'times': [], 'acc': []},
        'rbf': {'times': [], 'acc': []}
    }

    print(f"\nBenchmarking with {n_trials} trials per size...")
    print(f"{'Size':<10} | {'Linear (s)':<12} | {'RBF (s)':<12} | {'Lin Acc':<10} | {'RBF Acc':<10}")
    print("-" * 65)

    for n in subset_sizes:
        # Subsample training data
        X_sub_raw = X_train_full[:n]
        y_sub = y_train_full[:n]
        
        """
        we calculate the scaler on the full training data(60k samples) and then 
        transform each of the subsets using the same scaler
        """
        X_sub = scaler.transform(X_sub_raw)

        # --- Linear SVM ---
        lin_time, lin_acc = run_benchmark(
            lambda: LinearSVC(dual=False, C=1.0, max_iter=5000, tol=1e-2),
            X_sub, y_sub, X_test, y_test, num_trials=n_trials
        )
        results['linear']['times'].append(lin_time)
        results['linear']['acc'].append(lin_acc)

        # --- RBF SVM ---
        rbf_time, rbf_acc = run_benchmark(
            lambda: SVC(kernel='rbf', C=1.0, gamma='scale', tol=1e-2),
            X_sub, y_sub, X_test, y_test, num_trials=n_trials
        )
        results['rbf']['times'].append(rbf_time)
        results['rbf']['acc'].append(rbf_acc)

        print(f"{n:<10} | {lin_time:.4f}       | {rbf_time:.4f}       | {lin_acc:.4f}     | {rbf_acc:.4f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Linear Scale Plot (Time)
    ax1.set_title("Training Time vs Data Size (Linear Scale)")
    ax1.set_xlabel('Training Set Size (N)')
    ax1.set_ylabel('Time (seconds)')
    ax1.plot(subset_sizes, results['linear']['times'], 'o--', label='LinearSVC (O(N))')
    ax1.plot(subset_sizes, results['rbf']['times'], 'o-', label='RBF SVC (O(N^2)-O(N^3))')
    ax1.grid(True)
    ax1.legend()

    # 2. Log-Log Plot (Complexity Analysis)
    ax2.set_title("Complexity Analysis (Log-Log Scale)")
    ax2.set_xlabel('Log10(Training Set Size)')
    ax2.set_ylabel('Log10(Time)')
    ax2.loglog(subset_sizes, results['linear']['times'], 'o--', label='LinearSVC')
    ax2.loglog(subset_sizes, results['rbf']['times'], 'o-', label='RBF SVC')
    
    # Add theoretical slopes for reference
    # Normalize reference lines to start at the first data point
    ref_x = np.array(subset_sizes)
    
    # Linear O(N) reference
    # y = k * x  => log y = log k + log x.  Slope = 1
    # Match starting point of LinearSVC
    y_lin_ref = results['linear']['times'][0] * (ref_x / ref_x[0])**1.0
    ax2.loglog(ref_x, y_lin_ref, 'k:', alpha=0.5, label='Reference O(N)')
    
    # Quadratic O(N^2) reference
    # Match starting point of RBF
    y_rbf_ref = results['rbf']['times'][0] * (ref_x / ref_x[0])**2.0
    ax2.loglog(ref_x, y_rbf_ref, 'r:', alpha=0.5, label='Reference O(N^2)')
    
    ax2.grid(True, which="both", ls="-")
    ax2.legend()

    plt.tight_layout()
    plot_filename = 'svm_benchmark_plot.png'
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")

if __name__ == "__main__":
    main()