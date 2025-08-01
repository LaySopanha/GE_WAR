# CHES 2025 Competition - Quick Setup Guide

## 🚀 **Ready-to-Run Configuration (95% Confidence)**

This repository contains the **final optimized configuration** for CHES 2025 with:
- ✅ **Fixed JSON serialization** (no more corruption!)
- ✅ **95% confidence sweep parameters** based on comprehensive analysis
- ✅ **Attack-driven training** with composite scoring (1000000 * GE + NTGE)
- ✅ **350k trace optimization** (matching challenge organizer requirements)

## 📋 **Quick Start**

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
- Download CHES_2025 dataset to `../dataset/CHES_2025/` folder
- Ensure you have `CHES_Challenge.h5` file

### 3. Run the Optimized Sweep
```bash
# Start the sweep (same ID as original successful run)
wandb agent panhalay69420-idri/GE_WAR/hgfzx5rs
```

**Or create new sweep:**
```bash
wandb sweep sweep.yaml
# Then run: wandb agent <new_sweep_id>
```

## 🎯 **What to Expect**

- **GE=0 Target**: Models achieving GE=0 will be auto-saved with metadata
- **NTGE Optimization**: Lower NTGE = better model for same GE=0
- **Robust Saving**: No more crashes - all files saved properly
- **350k Traces**: Optimized for challenge evaluation size

## 📁 **Key Files**

- `main_pytorch.py` - Fixed training pipeline with robust saving
- `sweep.yaml` - 95% confidence hyperparameter configuration  
- `src/` - Core modules (CNN, trainer, dataloader, utils)
- `requirements.txt` - All dependencies

## 🏆 **Success Metrics**

- **Primary Goal**: GE = 0 
- **Secondary Goal**: Minimize NTGE (Number of Traces to GE=0)
- **Composite Score**: 1000000 * GE + NTGE (minimize this)

## 🔧 **Configuration Details**

**Proven Successful Parameters:**
- Learning Rate: [0.00005, 0.0001, 0.00002]
- Batch Size: [32, 64] 
- POI: [200, 250, 300]
- Epochs: 250
- Architecture: CNN with 3-4 conv layers, 16-32 filters

**Fixed Parameters:**
- Dataset: CHES_2025
- Leakage: ID (Identity) 
- Train Size: 350,000 traces
- Attack Size: 100,000 traces
- K-Folds: 2

Good luck! 🚀
