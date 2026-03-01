import time
import torch
from torchvision import datasets

from mlp import MLP_GD
from cnn import CNN_GD


def set_seed(seed: int = 42):
    torch.manual_seed(seed)


def main():
    set_seed(42)

    ###########################################################
    # Import Data
    ###########################################################

    ### MNIST ###
    MNIST_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True
    )
    MNIST_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True
    )

    # Normalize MNIST values
    MNIST_train_X = MNIST_train.data.float() / 255.0
    MNIST_test_X = MNIST_test.data.float() / 255.0

    m_mean = 0.13066047430038452
    m_std = 0.30810782313346863

    MNIST_train_X = ((MNIST_train_X - m_mean) / m_std).unsqueeze(1)
    MNIST_test_X = ((MNIST_test_X - m_mean) / m_std).unsqueeze(1)

    MNIST_train_y = MNIST_train.targets.long()
    MNIST_test_y = MNIST_test.targets.long()

    MNIST_train_X = MNIST_train_X[:50000]
    MNIST_train_y = MNIST_train_y[:50000]

    ### CIFAR-10 ###
    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )
    CIFAR10_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True
    )

    # Normalize CIFAR-10 pixels
    CIFAR10_train_X = torch.from_numpy(CIFAR10_train.data).float() / 255.0
    CIFAR10_train_X = CIFAR10_train_X.permute(0, 3, 1, 2)

    CIFAR10_test_X = torch.from_numpy(CIFAR10_test.data).float() / 255.0
    CIFAR10_test_X = CIFAR10_test_X.permute(0, 3, 1, 2)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    CIFAR10_train_X = (CIFAR10_train_X - mean) / std
    CIFAR10_test_X = (CIFAR10_test_X - mean) / std

    CIFAR10_train_y = torch.tensor(CIFAR10_train.targets).long()
    CIFAR10_test_y = torch.tensor(CIFAR10_test.targets).long()

    ###########################################################
    # Training Models
    ###########################################################

    # -------------------------
    # MLP on MNIST
    # -------------------------
    mlp = MLP_GD(input_size=1 * 28 * 28, hidden_size=128, output_size=10)

    t0 = time.perf_counter()
    acc_mlp_mnist = mlp.train(
        MNIST_train_X, MNIST_train_y,
        MNIST_test_X, MNIST_test_y,
        epochs=30, learning_rate=0.1, batch_size=128
    )
    t1 = time.perf_counter()
    print(f"[MLP MNIST] Training time: {t1 - t0:.2f} seconds\n")

    # -------------------------
    # CNN on MNIST
    # -------------------------
    cnn = CNN_GD(
        channels=1,
        img_size=28,
        num_classes=10,
        num_conv=2,
        num_fc=1,
        num_filters=32,
        pool_size=2
    )

    t0 = time.perf_counter()
    acc_cnn_mnist = cnn.train_model(
        MNIST_train_X, MNIST_train_y,
        MNIST_test_X, MNIST_test_y,
        epochs=30, lr=0.001, batch_size=128, l2_param=1e-3
    )
    t1 = time.perf_counter()
    print(f"[CNN MNIST] Training time: {t1 - t0:.2f} seconds\n")

    # -------------------------
    # MLP on CIFAR-10
    # -------------------------
    mlp_cifar10 = MLP_GD(input_size=3 * 32 * 32, hidden_size=128, output_size=10)

    t0 = time.perf_counter()
    acc_mlp_cifar10 = mlp_cifar10.train(
        CIFAR10_train_X, CIFAR10_train_y,
        CIFAR10_test_X, CIFAR10_test_y,
        epochs=30, learning_rate=0.1, batch_size=128
    )
    t1 = time.perf_counter()
    print(f"[MLP CIFAR-10] Training time: {t1 - t0:.2f} seconds\n")

    # -------------------------
    # CNN on CIFAR-10
    # -------------------------
    cnn_cifar10 = CNN_GD(
        channels=3,
        img_size=32,
        num_classes=10,
        num_conv=2,
        num_fc=1,
        num_filters=32,
        pool_size=2
    )

    t0 = time.perf_counter()
    acc_cnn_cifar10 = cnn_cifar10.train_model(
        CIFAR10_train_X, CIFAR10_train_y,
        CIFAR10_test_X, CIFAR10_test_y,
        epochs=30, lr=0.001, batch_size=128, l2_param=1e-3
    )
    t1 = time.perf_counter()
    print(f"[CNN CIFAR-10] Training time: {t1 - t0:.2f} seconds\n")


if __name__ == "__main__":
    main()