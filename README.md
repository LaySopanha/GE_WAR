# CHES 2025 Challenge Submission - GE_WAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2.5-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)
![CHES](https://img.shields.io/badge/CHES-2025-green)
![Status](https://img.shields.io/badge/GE-0.0-success)
![DGIST](https://img.shields.io/badge/Research-DGIST-purple)

## Overview

This repository contains **Submission #12** for the **CHES 2025 Side-Channel Analysis Challenge** by team **Ott3rly Av3rag3**. The project implements a deep learning-based approach using PyTorch for side-channel attack analysis on AES implementations.

This submission achieved the best performance on the **Private1** test set among all team submissions.

## Team Members

- **Lay Sopanha** - panhalay69420@gmail.com
- **Pav Limseng** - e20211548@dtc1.itc.edu.kh
- **Kao Vichet** - kaovichet11@gmail.com

## Advisor

- **Cho Seunghyun** - seunghyuncho@dgist.ac.kr

## Results

### Submission #12 Performance

- **Guessing Entropy (GE)**: 0.0
- **Number of Traces to GE (NTGE)**: 99,774
- **Overall Score**: 122,516.50

### Detailed Performance Across Test Sets

| Test Set  | NTGE      |
|-----------|-----------|
| Public    | 99,774    |
| Private1  | 99,993    |
| Private2  | 200,033   |
| Private3  | 90,266    |

### All Team Submissions Comparison

| Submission | Public  | Private1 | Private2 | Private3 | Score       |
|------------|---------|----------|----------|----------|-------------|
| #3         | 97,217  | 200,009  | 200,071  | 98,288   | 148,896.25  |
| #4         | 97,412  | 200,001  | 200,037  | 87,112   | 146,140.50  |
| #5         | 91,454  | 200,005  | 200,043  | 79,337   | 142,709.75  |
| #6         | 98,863  | 200,002  | 200,022  | 96,120   | 148,751.75  |
| #7         | 98,770  | 200,011  | 200,075  | 94,759   | 148,403.75  |
| #8         | 96,431  | 200,009  | 200,077  | 99,992   | 149,127.25  |
| #9         | 99,106  | 200,002  | 200,029  | 86,531   | 146,417.00  |
| #10        | 97,684  | 200,001  | 96,134   | 87,133   | 120,238.00  |
| #11        | 97,222  | 200,004  | 200,005  | 96,527   | 148,439.50  |
| **#12**    | **99,774** | **99,993** | **200,033** | **90,266** | **122,516.50** |

★ **Key Achievement**: Submission #12 demonstrated exceptional performance on the Private1 test set (99,993 traces), significantly outperforming all other submissions which required 200,000+ traces.

## Project Structure

```
.
├── analyze_pytorch.py          # Main analysis script
├── ge0_config_run_12.npy      # Model configuration
├── ge0_metadata_run_12.json   # Training metadata
├── ge0_model_run_12.pth       # Trained model weights
├── requirements.txt            # Python dependencies
├── submission.md               # Submission details
└── src/
    ├── dataloader.py          # Custom data loading utilities
    ├── net.py                 # Neural network architectures
    ├── trainer.py             # Training pipeline
    └── utils.py               # Utility functions
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd GE_WAR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (refer to [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific setup):
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Run the analysis script to perform side-channel attack evaluation:

```bash
python analyze_pytorch.py
```

The script will:
- Load the pre-trained model from `ge0_model_run_12.pth`
- Evaluate the model on the CHES 2025 dataset
- Calculate guessing entropy and key recovery performance

## Model Architecture

The implementation supports multiple neural network architectures:
- ▪ **MLP**: Multi-Layer Perceptron with configurable depth and activation functions
- ▪ **CNN**: Convolutional Neural Network optimized for trace analysis

### Supported Leakage Models

- ▪ **Identity (ID)**: Direct S-box output (256 classes)
- ▪ **Hamming Weight (HW)**: Hamming weight of S-box output (9 classes)

## Key Features

- ✓ Reproducible results with fixed random seeds
- ✓ Support for both CPU and GPU execution
- ✓ Configurable attack parameters
- ✓ Efficient data loading with custom PyTorch datasets
- ✓ Comprehensive evaluation metrics

## Dependencies

```
h5py==3.13.0
joblib==1.5.0
numpy==2.2.5
scikit-learn==1.6.1
scipy==1.15.3
threadpoolctl==3.6.0
tqdm==4.67.1
torch (see PyTorch installation)
torchvision (see PyTorch installation)
```

## References

- **CHES 2025 Challenge**: [https://pace-tl.gitbook.io/ches-challenge-2025](https://pace-tl.gitbook.io/ches-challenge-2025)
- **PACL Lab**: [https://sites.google.com/view/pacl/](https://sites.google.com/view/pacl/)

## License

This project is submitted as part of the CHES 2025 Challenge. Please refer to the challenge guidelines for usage and distribution terms.

## Acknowledgments

This work was developed as part of the **Summer Research Internship at DGIST** (Daegu Gyeongbuk Institute of Science and Technology) and submitted to the CHES 2025 Side-Channel Analysis Challenge organized by the PACL (Power Analysis and Cryptography Lab).

---

★ **Note**: This submission achieved a guessing entropy of 0.0, indicating successful key recovery with the provided model and configuration.
