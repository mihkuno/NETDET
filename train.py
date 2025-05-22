import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import argparse

# Custom dataset class for NSL-KDD
class NSLKDDDataset(Dataset):
    def __init__(self, file_path, encoders=None, scaler=None, fit=True):
        # Column names for NSL-KDD (41 features + label + difficulty)
        col_names = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
            'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
            'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
            'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
            'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
            'label','difficulty'
        ]
        # Read data with explicit separator and no header
        df = pd.read_csv(file_path, names=col_names, sep=',', header=None)
        # Separate features and labels
        X = df.drop(['label', 'difficulty'], axis=1)
        y = df['label']
        # Fit or transform encoders/scaler
        if fit:
            # Encode categorical features
            self.le_protocol = LabelEncoder()
            self.le_service = LabelEncoder()
            self.le_flag = LabelEncoder()

            for col in ['protocol_type', 'service', 'flag']:
                if col not in X.columns:
                    raise ValueError(f"Missing expected categorical column: {col}")

            X['protocol_type'] = self.le_protocol.fit_transform(X['protocol_type'].astype(str))
            X['service'] = self.le_service.fit_transform(X['service'].astype(str))
            X['flag'] = self.le_flag.fit_transform(X['flag'].astype(str))

            # Convert all to float
            try:
                X = X.astype(float)
            except Exception as e:
                print("Failed to cast features to float. Problematic data:")
                print(X.head())
                raise e

            # Save encoders dict
            self.encoders = {
                'protocol': self.le_protocol,
                'service': self.le_service,
                'flag': self.le_flag
            }
            # Scale numerical features
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            # Encode labels (multiclass)
            self.le_label = LabelEncoder()
            y = self.le_label.fit_transform(y)
        else:
            # Use provided encoders and scaler
            self.le_protocol = encoders['protocol']
            self.le_service = encoders['service']
            self.le_flag = encoders['flag']

            X['protocol_type'] = self.le_protocol.transform(X['protocol_type'].astype(str))
            X['service'] = self.le_service.transform(X['service'].astype(str))
            X['flag'] = self.le_flag.transform(X['flag'].astype(str))

            X = X.astype(float)
            X = scaler.transform(X)
            y = encoders['label'].transform(y)
            self.encoders = encoders
            self.scaler = scaler
        # Store data
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define neural network model
class NSLModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NSLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Training function
def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset = NSLKDDDataset(args.train_file, fit=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    encoders = train_dataset.encoders
    encoders['label'] = train_dataset.le_label
    scaler = train_dataset.scaler
    input_dim = train_dataset.X.shape[1]
    num_classes = len(train_dataset.le_label.classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NSLModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    all_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_dataset)
        all_losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{args.epochs}], Loss: {epoch_loss:.4f}")

    torch.save({'model_state_dict': model.state_dict(),
                'encoders': encoders,
                'scaler': scaler},
               os.path.join(args.output_dir, 'nsl_model.pth'))

    plt.figure()
    plt.plot(range(1, args.epochs + 1), all_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))
    print("Training complete. Model and plot saved.")
    print("Training Summary:")
    print(f"Total epochs: {args.epochs}")
    print(f"Final Loss: {all_losses[-1]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NSL-KDD multiclass model')
    parser.add_argument('--train_file', type=str, required=True, help='Path to KDDTrain+.txt')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and plots')
    args = parser.parse_args()
    train(args)
    
    # Example:
    # python train.py --train_file dataset/kddtrain.csv --epochs 30 --batch_size 64 --lr 0.001 --output_dir ./output