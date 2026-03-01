import torch

class MLP_GD:

    def __init__(self, input_size, hidden_size, output_size, seed: int = 42):
        torch.manual_seed(seed)

        # Parameters
        self.w1 = torch.randn(input_size, hidden_size) * 0.01 # Random initialisation
        self.w2 = torch.randn(hidden_size, output_size) * 0.01
        self.b1 = torch.zeros((1, hidden_size))
        self.b2 = torch.zeros((1, output_size))


    ### Activation functions ###
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def softmax(self, x): # for last layer
        x = x - x.max(dim=1, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=1, keepdim=True)

    ### Forward Pass ###
    def forward(self, X):
        assert X.dim() == 4, f"Expected input (N,C,H,W), got {X.shape}"
        X_flat = X.reshape(X.shape[0], -1)
        self.X_flat = X_flat # for backpropogation

        self.hidden_input = X_flat @ self.w1 + self.b1
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = self.hidden_output @ self.w2 + self.b2
        self.final_output = self.softmax(self.final_input)

        return self.final_output

    
    ### Backpropogation ###
    def backward(self, y, output, lr):

        N = y.shape[0]

        grad_logits = output.clone()
        grad_logits[torch.arange(N), y] -= 1.0
        grad_logits /= N

        hidden_error = (grad_logits @ self.w2.T) * self.hidden_output * (1 - self.hidden_output)

        # Gradient step
        self.w2 -= lr * self.hidden_output.T @ grad_logits
        self.b2 -= lr * grad_logits.sum(dim=0, keepdim=True)

        self.w1 -= lr * self.X_flat.T @ hidden_error
        self.b1 -= lr * hidden_error.sum(dim=0, keepdim=True)

    ### Training cycle ###
    def train(self, X_train, y_train, X_test, y_test, # test sets only for printing accuracy
              epochs, learning_rate, batch_size): ## Hyperparameters

        train_accuracies = []
        test_accuracies = []
        n = X_train.shape[0]

        # Initial accuracies (model not trained)
        train_acc0 = self.accuracy(X_train, y_train)
        test_acc0 = self.accuracy(X_test, y_test)
        train_accuracies.append(train_acc0)
        test_accuracies.append(test_acc0)
        print(f"Epoch 0 | Train Acc: {train_acc0*100:.2f}% | Test Acc: {test_acc0*100:.2f}%")

        # Every Epoch:
        for epoch in range(epochs):
            idx = torch.randperm(n)

            # Mini-batching:
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                output_batch = self.forward(X_batch)
                self.backward(y_batch, output_batch, learning_rate)

            # Log-loss on training set
            output_train = self.forward(X_train).clamp_min(1e-9)
            loss = -torch.log(output_train[torch.arange(n), y_train]).mean()
            
            train_acc = self.accuracy(X_train, y_train)
            test_acc = self.accuracy(X_test, y_test)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Print current model accuracy
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

        return train_accuracies, test_accuracies

    
    def predict(self, X):
        output = self.forward(X)
        return torch.argmax(output, dim=1)

    def accuracy(self, X, y_true):
        y_true = y_true.long().view(-1)
        with torch.no_grad():
            y_pred = self.predict(X)
            correct = (y_pred == y_true).sum()
            return (correct.float() / y_true.shape[0]).item()