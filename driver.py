import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
print(script_directory)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Plot in real-time
class LiveCurve:
    def __init__(self, title="Learning Curve", ylabel="Loss"):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        (self.train_line,) = self.ax.plot([], [], label="Train")
        (self.dev_line,)   = self.ax.plot([], [], label="Dev")
        self.train_hist, self.dev_hist = [], []
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.legend()
        self.fig.show()          # non-blocking
        self.fig.canvas.draw()

    def update(self, train_loss, dev_loss):
        self.train_hist.append(train_loss)
        self.dev_hist.append(dev_loss)
        xs = range(1, len(self.train_hist) + 1)
        self.train_line.set_data(xs, self.train_hist)
        self.dev_line.set_data(xs, self.dev_hist)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # allow GUI to refresh

## Dataset for loading and preprocessing the COVID-19 dataset
class COVID19dataset(Dataset):
    def __init__(self, file_path, mode='train', target_only=False, normalize=True):
        self.mode = mode

        # 1. Read the csv file into numpy array
        df = pd.read_csv(file_path, index_col=0)
        data = df.to_numpy(dtype=np.float32)
        # 2. Split the data into features and labels
        if not target_only:
            feats = list(range(data.shape[1]-1))  # All columns except the last one
        else:
            ''' Compute the correlation matrix of the features '''
            cor = df.corr()['tested_positive.2']
            best_features = cor[cor.abs() > 0.7].index.tolist()
            feats = list(range(40)) + [df.columns.get_loc(feat) for feat in best_features]
            # print(f"Selected features based on correlation > 0.5 with target: {best_features}")
            # print(f"Feature indices: {feats}")
            
        
        if mode == 'test':
            # 2.1 For test mode, use the last column as labels
            # but we don't need labels in test set, so just return features
            data = data[:, feats]
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            # 2.2 For train/val mode, split the data into features and labels
            data = data[:, feats]
            labels = data[:, -1]  # Assuming the last column is the label

            # 2.3 Splitting training data into train & dev sets
            indices = np.arange(data.shape[0]) # create shuffled indices
            np.random.shuffle(indices)
            split = int(0.9 * len(indices)) # 90/10 split
            if mode == 'train':
                idx = indices[:split]
            elif mode == 'dev':
                idx = indices[split:]

            # separate features and labels
            self.data = torch.tensor(data[idx], dtype=torch.float32)   # X
            self.labels = torch.tensor(labels[idx], dtype=torch.float32)    # y

        # 3. Normalize the features
        if normalize:
            mean = self.data[:, 40:].mean(dim=0, keepdim=True)
            std = self.data[:, 40:].std(dim=0, keepdim=True)
            self.data[:, 40:] = (self.data[:, 40:] - mean) / std

        self.dim = self.data.shape[1]  # Number of features
        
        print(f'Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.data.shape[1]})')
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''The __getitem__ function loads and returns a sample from the dataset at the given index idx.'''
        if self.mode == 'test':
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]

## Deep Neural Network Model
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits.squeeze(1)
    
def prep_dataLoader(path, mode, target_only=False, normalize=True, num_workers=0, batch_size=32, shuffle=True):
    dataset = COVID19dataset(file_path=path, mode=mode, target_only=target_only, normalize=normalize)
    dataLoader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers
                            )
    return dataLoader

def train_loop(dataloader, model, loss_fn, optimizer, L1 = 0.0, L2 = 0.0):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        data_loss = loss_fn(pred, y)  # this is a mean over batch

        # L1/L2 regularization (skip biases/Norm params) 
        lambda_l1, lambda_l2 = L1, L2

        if lambda_l1 > 0.0 or lambda_l2 > 0.0:
            l1 = torch.zeros((), device=pred.device)
            l2 = torch.zeros((), device=pred.device)
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                # commonly exclude biases and norm parameters
                if ('bias' in name) or any(s in name.lower() for s in ('bn','batchnorm','ln','layernorm','norm')):
                    continue
                if lambda_l1 > 0.0:
                    l1 = l1 + p.abs().sum()
                if lambda_l2 > 0.0:
                    l2 = l2 + p.square().sum()  # same as p.pow(2).sum()

            # scale reg to match mean reduction
            loss = data_loss + (lambda_l1 * l1 + lambda_l2 * l2) / X.size(0)
        else:
            loss = data_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"\tTrain MSE loss: {loss:>8f} \n")
        return loss.detach().cpu().item()

def val_loop(dataloader, model, loss_fn):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dataloader:                         # iterate through the dataloader
        x, y = x.to(DEVICE), y.to(DEVICE)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = loss_fn(pred, y)         # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dataloader.dataset)              # compute averaged loss
    print(f"\tValidation MSE loss: {total_loss:>8f} \n")
    return total_loss

def plot_pred(dataloader, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred = model(X)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    plt.ioff()
    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


if __name__ == '__main__':
    # Hyperparameters ##############################
    epochs = 1000
    batch_size = 270
    learning_rate = 1e-2
    early_stopping_patience = 200
    Target_only = True  # whether to use all features or only those highly correlated with the target
    L1_reg = 0.0
    L2_reg = 0.0
    ################################################

    train_set = prep_dataLoader(path=script_directory + r'/Data/covid.train.csv', mode='train', target_only=Target_only, normalize=True, num_workers=0, batch_size=batch_size, shuffle=True)
    dev_set = prep_dataLoader(path=script_directory + r'/Data/covid.train.csv', mode='dev', target_only=Target_only, normalize=True, num_workers=0, batch_size=batch_size, shuffle=False)
    test_set = prep_dataLoader(path=script_directory + r'/Data/covid.test.csv', mode='test', target_only=False, normalize=True, num_workers=0, batch_size=batch_size, shuffle=False)

    live = LiveCurve(title="COVID DNN Learning Curve", ylabel="MSE")
    
    model = DNN(train_set.dataset.dim, 1).to(DEVICE)  # Construct model and move to device
    loss_fn = nn.MSELoss(reduction='mean')  # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stop_cnt = 0

    history = {"train_loss": [], "dev_loss": []}
    min_mse = float('inf')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_set, model, loss_fn, optimizer, L1 = L1_reg, L2 = L2_reg)
        dev_mse = val_loop(dev_set, model, loss_fn)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_mse)
        live.update(train_loss, dev_mse)
        if dev_mse < min_mse:
            min_mse = dev_mse
            torch.save(model.state_dict(), script_directory + r'/Model/covid_dnn.pth')
            print(f"Model saved with MSE {min_mse:>8f}")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        if early_stop_cnt >= early_stopping_patience:
            print("Early stopping triggered!")
            break

    print("Done!")

    del model                               # delete the model
    model = DNN(train_set.dataset.dim, 1).to(DEVICE)  # re-instantiate the model
    model.load_state_dict(torch.load(script_directory + r'/Model/covid_dnn.pth'))  # load the best model
    plot_pred(dev_set, model, DEVICE)  # plot prediction of dev set

