import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# Custom dataset for test data
class NSLKDDTestDataset(Dataset):
    def __init__(self, file_path, encoders, scaler):
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
        df = pd.read_csv(file_path, names=col_names, sep=',', header=None)

        known_labels = set(encoders['label'].classes_)
        unknown_labels = set(df['label']) - known_labels
        if unknown_labels:
            print(f"[WARN] Skipping {len(unknown_labels)} unknown labels in test set: {unknown_labels}")

        df = df[df['label'].isin(known_labels)].copy()

        X = df.drop(['label', 'difficulty'], axis=1)
        y = df['label']

        X['protocol_type'] = encoders['protocol'].transform(X['protocol_type'].astype(str))
        X['service'] = encoders['service'].transform(X['service'].astype(str))
        X['flag'] = encoders['flag'].transform(X['flag'].astype(str))
        X = X.astype(float)
        X = scaler.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(encoders['label'].transform(y), dtype=torch.long)
        self.raw_y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural network model (same as training)
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

# Function to load model, encoders, and scaler once
def load_model_components(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    encoders = checkpoint['encoders']
    scaler = checkpoint['scaler']

    input_dim = len(checkpoint['scaler'].mean_)  # Number of features expected
    num_classes = len(encoders['label'].classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NSLModel(input_dim, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, encoders, scaler, device

# Predict single sample function (input string without label)
def predict_single_sample(input_str, model, encoders, scaler, device):
    col_names_no_label = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
        'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
        'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
    ]

    # Split input string and convert to dataframe with one row
    values = input_str.strip().split(',')
    if len(values) != len(col_names_no_label):
        raise ValueError(f"Input sample length {len(values)} does not match expected {len(col_names_no_label)} features.")

    df = pd.DataFrame([values], columns=col_names_no_label)

    # Apply label encoders on categorical features
    df['protocol_type'] = encoders['protocol'].transform(df['protocol_type'].astype(str))
    df['service'] = encoders['service'].transform(df['service'].astype(str))
    df['flag'] = encoders['flag'].transform(df['flag'].astype(str))

    df = df.astype(float)

    # Scale
    X_scaled = scaler.transform(df)

    # Convert to tensor and send to device
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_index = np.argmax(probs)
        pred_label = encoders['label'].classes_[pred_index]
        pred_prob = probs[pred_index]

    return pred_label, pred_prob

# Inference function
def inference(args):
    model, encoders, scaler, device = load_model_components(args.model_path)

    test_dataset = NSLKDDTestDataset(args.test_file, encoders, scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("Evaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    normal_index = np.where(encoders['label'].classes_ == 'normal')[0][0]
    print("\nSample Normal vs Attack Probabilities (first 10 samples):")
    for i in range(min(10, len(all_probs))):
        prob_normal = all_probs[i][normal_index]
        prob_attack = 1.0 - prob_normal
        highest_prob = max(all_probs[i])
        highest_label_index = np.argmax(all_probs[i])
        highest_label = encoders['label'].classes_[highest_label_index]
        print(
            f"Sample {i+1}: Normal: {prob_normal*100:.2f}% | Attack: {prob_attack*100:.2f}% | "
            f"Predicted Label: {highest_label} ({highest_prob*100:.2f}%)"
        )

    # Additional: Accuracy by attack type
    inv_label_encoder = {i: label for i, label in enumerate(encoders['label'].classes_)}
    pred_labels = [inv_label_encoder[i] for i in all_preds]
    true_labels = [inv_label_encoder[i] for i in all_labels]

    df_results = pd.DataFrame({'y_true': true_labels, 'y_pred': pred_labels})
    df_results['attack_type'] = df_results['y_true'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    df_results['predicted_type'] = df_results['y_pred'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    attack_type_acc = accuracy_score(df_results['attack_type'], df_results['predicted_type'])
    print(f"\nAccuracy in classifying as Normal vs Attack: {attack_type_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSL-KDD Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model .pth file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to KDDTest+.txt')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    args = parser.parse_args()

    # Load model components once
    model, encoders, scaler, device = load_model_components(args.model_path)

    # Run batch inference on test set
    inference(args)
    
    # Example:
    # python inference.py --model_path output/nsl_model.pth --test_file dataset/kddtest.csv --batch_size=64