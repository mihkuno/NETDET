# 🛡️ NSL-KDD Neural Network

A deep learning approach for network intrusion detection using the NSL-KDD dataset. This neural network model can classify network traffic as normal or various types of attacks.

## 🚀 Quick Start

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows** 🪟

```bash
./venv/Scripts/activate
```

**Linux/macOS** 🐧

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Training & Testing (Optional) ⚙️

You can skip this step if you just want to run the main application.

**Training** 🏋️‍♂️

```bash
python train.py --train_file dataset/kddtrain.csv --epochs 30 --batch_size 64 --lr 0.001 --output_dir ./output
```

**Testing** 🧪

```bash
python inference.py --model_path output/nsl_model.pth --test_file dataset/kddtest.csv --batch_size=64
```

### Step 5: Run the Application 🎯

```bash
# Edit main.py to configure your settings
python main.py
```

## 📁 Project Structure

```
├── dataset/
│   ├── kddtrain.csv
│   └── kddtest.csv
├── output/
│   └── nsl_model.pth
├── main.py
├── train.py
├── inference.py
└── requirements.txt
```

## 🔧 Configuration

Edit `main.py` to customize:

- Model parameters
- Input data paths
- Output settings
- Detection thresholds

## 📊 About NSL-KDD Dataset

The NSL-KDD dataset is an improved version of the original KDD Cup 1999 dataset, designed for network intrusion detection research. It contains various types of network attacks including:

- **DoS** (Denial of Service)
- **Probe** (Surveillance and probing)
- **R2L** (Remote to Local attacks)
- **U2R** (User to Root attacks)

## 🤖 Model Features

- Deep neural network architecture
- Multi-class classification
- Configurable hyperparameters
- Batch processing support
- Model checkpointing

## 📈 Performance

The model is trained to detect network intrusions with high accuracy while minimizing false positives for normal network traffic.

## 🛠️ Requirements

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Scikit-learn

## 📝 License

This project is open source and available under the MIT License.

---

Made with ❤️ for cybersecurity research
