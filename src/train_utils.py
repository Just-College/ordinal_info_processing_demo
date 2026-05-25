import csv
import os
import time

import torch
from tqdm import tqdm


def train_model(model, loader_train, loader_test, optimizer, scheduler, loss_fn, device, epochs, csv_dir, model_dir, params_str):
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(loader_train, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=False, leave=True):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / len(loader_train)
        # 测试集评估
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader_test:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                test_loss += loss.item()
                # 只在音素区间评估准确率
                # 第一个音素区间
                pred1 = out[:, 30 : 30 + 3, :].mean(dim=1).argmax(dim=1)
                target1 = y[:, 30 : 30 + 3, :].mean(dim=1).argmax(dim=1)
                # 第二个音素区间
                pred2 = out[:, 30 + 3 + 30 : 30 + 3 + 30 + 3, :].mean(dim=1).argmax(dim=1)
                target2 = y[:, 30 + 3 + 30 : 30 + 3 + 30 + 3, :].mean(dim=1).argmax(dim=1)
                correct += (pred1 == target1).sum().item()
                correct += (pred2 == target2).sum().item()
                total += 2 * x.size(0)
        test_loss_avg = test_loss / len(loader_test)
        test_acc = correct / total * 100
        print(f"Epoch {epoch+1} loss: {train_loss:.4f}, test_loss: {test_loss_avg:.4f}, test_acc: {test_acc:.2f}%, lr: {scheduler.get_last_lr()[0]:.6f}")
        history.append([epoch + 1, train_loss, test_loss_avg, test_acc, scheduler.get_last_lr()[0]])
    # 保存训练过程到csv
    csv_path = os.path.join(csv_dir, f"train_history_{params_str}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "test_acc", "lr"])
        writer.writerows(history)
    print(f"训练过程已保存到: {csv_path}")
    # 保存模型
    os.makedirs(model_dir, exist_ok=True)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"rnn_{params_str}_{current_time}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
