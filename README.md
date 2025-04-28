# AgingAOE-Net

AgingAOE-Net is a deep learning framework for **Age-related Omics Exploration**, designed to analyze multi-omics data (e.g., genomics, transcriptomics, proteomics) to uncover aging-related biological insights. This repository provides the implementation, including scripts for data preprocessing, training, evaluation, and inference.

## Table of Contents

- Overview
- Features
- Installation
- Dataset
- Usage
  - Preprocessing
  - Training
  - Evaluation
  - Inference
- Model Architecture
- Contributing
- License
- Citation
- Contact

## Overview

AgingAOE-Net integrates multi-omics data to study aging mechanisms, identifying biomarkers and patterns for applications in biological research and personalized medicine. The repository includes code for model training, evaluation, and data preprocessing, as described in the associated research.

## Features

- **Multi-omics Integration**: Processes diverse omics data types.
- **Flexible Architecture**: Customizable neural network configurations.
- **Preprocessing Tools**: Scripts to format raw omics data.
- **Training & Evaluation**: Supports training and performance metrics.
- **Inference**: Predicts age-related outcomes with pre-trained models.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/FengAoWang/AgingAOE-Net.git
   cd AgingAOE-Net
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Requires:

   - Python 3.8+
   - PyTorch
   - NumPy
   - Pandas
   - Scikit-learn

## Dataset

AgingAOE-Net uses multi-omics data in CSV/TSV format. Datasets are available on Zenodo.

1. **Download data**:

   - Place datasets in the `data/` directory.
   - Follow the structure in `data/sample_data/` for custom data.

2. **Preprocessing**:

   ```bash
   python data/preprocess.py --input data/raw --output data/processed
   ```

## Usage

### Preprocessing

Format raw omics data:

```bash
python data/preprocess.py --input data/raw --output data/processed
```

### Training

Train the model:

```bash
python train.py --data_dir data/processed --output_dir outputs/ --epochs 50 --batch_size 32 --lr 0.001
```

- `--data_dir`: Processed data path
- `--output_dir`: Save model checkpoints/logs
- `--epochs`: Training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Evaluation

Evaluate the model:

```bash
python evaluate.py --model_path outputs/model.pth --data_dir data/processed
```

Reports metrics like MSE, accuracy, or AUC.

### Inference

Predict with a pre-trained model:

```bash
python infer.py --model_path outputs/model.pth --input data/processed/sample.csv
```

## Model Architecture

AgingAOE-Net features:

- **Input Layer**: Handles high-dimensional omics data.
- **Encoder**: Extracts features via convolutional/transformer layers.
- **Fusion Module**: Integrates multi-omics features.
- **Prediction Head**: Outputs aging-related predictions.

Details in `docs/model_architecture.md`.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push (`git push origin feature-branch`).
5. Open a pull request.

See CONTRIBUTING.md.

## License

Licensed under the MIT License. See LICENSE.

## Citation

Please cite the associated work if using AgingAOE-Net:

> \[Update with citation when available\]

Reference the dataset:

> Zenodo Dataset

## Contact

- **Maintainer**: Feng Ao Wang
- **GitHub**: FengAoWang
- **Issues**: GitHub Issues
