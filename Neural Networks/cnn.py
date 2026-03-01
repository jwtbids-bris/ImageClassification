import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_GD(nn.Module): ## Inherets from torch.nn.Module for optimisation
    
    def __init__(self, channels, img_size, num_classes,
                 num_conv, num_fc, num_filters, pool_size, seed = 42): # Hyperparameters
        
        super().__init__() # Inherit parent class properties
        torch.manual_seed(seed)

        # Store hyperparameters
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.num_filters = num_filters
        self.pool_size = pool_size

        self.convs = nn.ModuleList()
        self.pool = nn.MaxPool2d(pool_size)

        ## Initialise Conv layers
        for i in range(num_conv):
            in_ch = channels if i == 0 else num_filters
            self.convs.append(nn.Conv2d(in_ch, num_filters, kernel_size=3, padding=1))

        # Flatten dim after pooling each conv
        h = img_size // (pool_size ** num_conv)
        w = img_size // (pool_size ** num_conv)
        self.flat_dim = num_filters * h * w

        self.fcs = nn.ModuleList()
        in_dim = self.flat_dim
        for i in range(num_fc):
            out_dim = num_filters if i < num_fc - 1 else num_classes
            self.fcs.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

    ## Forward Pass ###
    def forward(self, X):

        # Conv Layers
        for conv in self.convs:
            X = F.relu(conv(X))
            X = self.pool(X)

        X = X.view(X.size(0), -1) # Flatten

        # FC Layers
        for i, fc in enumerate(self.fcs):
            X = fc(X)
            if i < len(self.fcs) - 1:
                X = F.relu(X)

        return X  # logits

    ### Training Cycle ###
    def train_model(self, X_train, y_train, X_test, y_test,
                    epochs, lr, batch_size, l2_param): # More hyperparams

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # uses gpu if possible
        self.to(device)

        optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_param) # Adam optimiser
        criterion = nn.CrossEntropyLoss()

        n = X_train.shape[0]
        train_accuracies, test_accuracies = [], []

        # Initial accuracies
        train_acc0 = self.accuracy(X_train, y_train)
        test_acc0 = self.accuracy(X_test, y_test)
        train_accuracies.append(train_acc0)
        test_accuracies.append(test_acc0)
        print(f"Epoch 0 | Train Acc: {train_acc0*100:.2f}% | Test Acc: {test_acc0*100:.2f}%")

        # Every epoch:
        for epoch in range(epochs):
            self.train() # training mode
            perm = torch.randperm(n)

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batching:
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                Xb = X_train[idx].to(device)
                yb = y_train[idx].to(device)

                # Foward and backward pass:
                optimiser.zero_grad()
                logits = self.forward(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimiser.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            # Update loss and accuaries:
            avg_loss = epoch_loss / max(num_batches, 1)
            train_acc = self.accuracy(X_train, y_train, batch_size=1000)
            test_acc  = self.accuracy(X_test, y_test, batch_size=1000)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Print output
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

        return train_accuracies, test_accuracies

    # Predict using trained model:
    def predict(self, X):
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(device))
            return torch.argmax(logits, dim=1).cpu()

    # Print accuracy from some true labels:
    def accuracy(self, X, y_true, batch_size=1000):
        device = next(self.parameters()).device
        self.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                xb = X[i:i+batch_size].to(device)
                yb = y_true[i:i+batch_size].to(device)
    
                logits = self.forward(xb)
                preds = torch.argmax(logits, dim=1)
    
                correct += (preds == yb).sum().item()
                total += yb.size(0)
    
        return correct / total