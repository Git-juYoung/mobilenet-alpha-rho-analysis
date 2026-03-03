import time
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    start = time.time()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    bar = tqdm(loader, desc=f"[{epoch}/{num_epochs}] Train", leave=False)

    for x, y in bar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (outputs.argmax(1) == y).sum().item()
        total_count += bs

        bar.set_postfix(loss=loss.item())

    train_time = time.time() - start
    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc, train_time


def evaluate_one_epoch(model, loader, criterion, device, epoch, num_epochs, mode):
    model.eval()
    start = time.time()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    bar = tqdm(loader, desc=f"[{epoch}/{num_epochs}] {mode}", leave=False)

    with torch.no_grad():
        for x, y in bar:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(1) == y).sum().item()
            total_count += bs

            bar.set_postfix(loss=loss.item())

    eval_time = time.time() - start
    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc, eval_time