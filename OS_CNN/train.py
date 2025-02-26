import torch
import torch.optim as optim
import torch.nn as nn
# from models import OSCNN
# from data_utils import get_dataloaders

def train_model_OSCNN(num_epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loader, test_loader = get_dataloaders(batch_size=32)

    # 初始化 OS-CNN 模型
    model = OSCNN(input_dim=1, num_classes=5).to(device)

    # 交叉熵损失 + Adam 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 训练完成后评估模型
    evaluate_model(model, test_loader)

    # 保存模型
    torch.save(model.state_dict(), "oscnn_model.pth")

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions)

    from sklearn.metrics import accuracy_score, classification_report
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))