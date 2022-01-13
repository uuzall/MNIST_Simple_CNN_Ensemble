import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def training_loop(model, n_epochs, model_name, train_dataloader):
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    correct, max_correct = 0, 0
    losses, accuracies, test_accuracies = list(), list(), list()

    for epoch in range(n_epochs):
        model.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        for batch_id, (x, y) in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            cat = torch.argmax(out, dim=1).to(device)
            acc = (cat == y).float().mean()
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            accuracies.append(acc.item())
            losses.append(loss.item())

            loop.set_description(f"Epoch: [{epoch+1} / {n_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=acc.item(), test=(correct/len(test)))

        model.eval()
        test_loss, correct = 0, 0
        total_pred, total_target = np.zeros(0), np.zeros(0)
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                total_pred = np.append(total_pred, pred.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            if max_correct < correct:
                torch.save(model.state_dict(), model_name)
                max_correct = correct
        test_accuracies.append((correct/len(test)*100))

        lr_scheduler.step()
    plt.plot(losses)
    plt.plot(accuracies)
    plt.plot(test_accuracies)

    return model
