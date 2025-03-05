import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

def train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, fine_tune=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if fine_tune:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

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

        avg_train_loss = running_loss / len(train_loader)

        if fine_tune and epoch == int(num_epochs * 0.5):
            for param in model.parameters():
                param.requires_grad = True

        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

def evaluate_model(model, test_loader):

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predictions)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()