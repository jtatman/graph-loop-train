import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Custom Dataset
# ---------------------------
class LayaDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col):
        self.X = torch.tensor(dataframe[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Example Laya-like Model
# ---------------------------
class LayaModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LayaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------
# Training Function
# ---------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                _, predicted = torch.max(outputs, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"Val Accuracy: {correct/total:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"laya_checkpoint_epoch{epoch+1}.pth")

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Laya-like model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.data)
    feature_cols = df.columns[:-1]  # all except last column
    target_col = df.columns[-1]

    # Encode target if categorical
    if df[target_col].dtype == object:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    # Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Data loaders
    train_loader = DataLoader(LayaDataset(train_df, feature_cols, target_col),
                               batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LayaDataset(val_df, feature_cols, target_col),
                             batch_size=args.batch_size)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayaModel(input_dim=len(feature_cols),
                      hidden_dim=args.hidden_dim,
                      output_dim=len(df[target_col].unique())).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

