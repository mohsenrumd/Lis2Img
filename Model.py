import torch
from torch import nn
import numpy as np
from tqdm.notebook import tqdm


class Lis2Img(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(64, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32,32, 4, stride=2, padding=2),
            nn.ReLU(),

        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 4, stride=2, padding=2)
        )

    def forward(self, x):
        # x has shape [4, 64, 1803]
        # # Copy embedding vector into all time points
        ########x = torch.cat((x, subj), dim=1)
        # x now has shape [4, 64+16, 1803]


        # Conv1d input shape: [batch size, # channels, signal length]
        x = self.encoder(x)

        # LSTM input shape: [batch size, signal length, # features] -- when batch_first=True!
        x = x.transpose(1, 2)
        # x, _ = self.lstm(x)
        x = x.transpose(2, 1)

        x = self.decoder(x)
        return x
    
device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device)) 
    
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def R_value(a, b):                      # Inputs shape [4, 64, 1803]
    dim = -1                            # Correlation over time points dimension
    a = a - a.mean(axis=dim, keepdim=True)
    b = b - b.mean(axis=dim, keepdim=True)
    cov = (a * b).mean(dim)
    na, nb = [(i**2).mean(dim)**0.5 for i in [a, b]]
    norms = na * nb
    R_matrix = cov / norms
    return R_matrix                     # [4, 64] - R values per channel for each trial


def train(x_dataloader, y_dataloader, model, loss_fn, optimizer):
    assert (len(x_dataloader.dataset) == len(y_dataloader.dataset)), \
                "Input, output dataloaders have different lengths! :O"
                
    # epoch_iter_in = tqdm_notebook(range(len(x_dataloader)), leave=False, total=len(x_dataloader), desc='Train'))
            
    
    size = len(x_dataloader.dataset)
    batch_idx=0
    for batch, (X, Y) in enumerate(zip(x_dataloader, y_dataloader)):
        batch+=1
        X, Y = X.to(device), Y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 1000 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"[{current:>4d}/{size:>4d}]   loss: {loss}")
        # epoch_iter_in.set_postfix(loss=loss / (batch_idx + 1))

def test(x_dataloader, y_dataloader, model, loss_fn, print_result = False):
    assert (len(x_dataloader) == len(y_dataloader)), \
                "Input, output dataloaders have different lengths! :O"

    num_batches = len(x_dataloader)
    model.eval()
    avg_loss = 0
    R_values = []

    with torch.no_grad():
        for (X, Y) in zip(x_dataloader, y_dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, Y).item()
            R = R_value(pred, Y)
            R_values.extend(R.flatten().tolist())

    avg_loss /= num_batches
    avg_R = np.asarray(R_values).mean()
    if print_result:
        print("Test:")
        print(f"\tAvg loss: {avg_loss}")
        print(f"\tAvg R value: {avg_R}\n")