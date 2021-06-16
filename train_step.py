import torch


def train(dataloader, model, loss_fn, optimizer, device):
    """[summary]

    Args:
        dataloader (): [description]
        model (nn.Model): NN model 
        loss_fn ([): Loss function to use during training step
        optimizer (): Type optimezer used during training step
        device (str): Device to place objects and operations
    """
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Error: \n Avg MSE: {test_loss:>8f} \n")
            