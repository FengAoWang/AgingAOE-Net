# AgingAOE-Net

AgingAOE-Net is a deep learning framework for **Age Order Enhanced Network**, designed to analyze multi-omics data (e.g., transcriptomics, DNA methylation) to uncover aging-related biological insights. This repository provides the implementation, including scripts for data preprocessing, training, evaluation, and downstream analysis.

![AgingAOE-Net Overview](figure1.png)

## Table of Contents

- Overview

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

   - Python 3.9+
   - PyTorch
   - NumPy
   - Pandas
   - Scikit-learn

## Dataset

AgingAOE-Net uses multi-omics data in CSV/TSV format. Datasets are available on Zenodo.

1. **Download data**:

   - Place datasets in the `data/` directory.
   - The all datasets used in this study are all avilable.


## Usage
We provide a demo of training and predicting the bio-age.

### Training

Pre-train the model:

```bash
cd train
python agingFound_pretraining.py 
```



### Bio-age prediction and PAAG calculation

finetune the model:

```bash
python agePredict.py 
```

