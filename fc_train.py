import torch
from torch import nn
from train_step import train, test
from data import TSData
from torch.utils.data import DataLoader
from fc_model import FullConnect

device = "cuda" if torch.cuda.is_available() else "cpu"
n_steps = 24

train_ds = DataLoader(TSData(7000, n_steps), 64)
test_ds = DataLoader(TSData(1000, n_steps), 64)

model = FullConnect(n_steps-1, 10, 1)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} \n -------------------------")
    train(train_ds, model, loss_fn, optimizer, device)
    test(test_ds, model, loss_fn, device)
print("Finished training")
