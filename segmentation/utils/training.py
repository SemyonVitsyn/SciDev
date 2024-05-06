import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

def plot_loss(losses):
    plt.plot(losses[0], label="train_loss")
    plt.plot(losses[1], label="val_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid()

    plt.show()

def train(model, criterion, optimizer, epochs, train_dataloader, valid_dataloader, device='cpu', scheduler=None, save=False, path=None):
    train_history = []
    valid_history = []

    for epoch in range(epochs):
        train_loss, valid_loss = 0, 0

        model.train()
        for X_batch, Y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()

            pred = model(X_batch)
            loss = criterion(pred, Y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss

        model.eval()
        for X_batch, Y_batch in valid_dataloader:
            with torch.no_grad():
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                pred = model(X_batch)
                loss = criterion(pred, Y_batch)

                valid_loss += loss

        if scheduler is not None:
            scheduler.step()
        
        train_history.append(train_loss.item() / len(train_dataloader))
        valid_history.append(valid_loss.item() / len(valid_dataloader))

        clear_output(True)
        plot_loss((train_history, valid_history))


    if save and path is not None:
        torch.save(model.state_dict(), path)


    return train_history, valid_history