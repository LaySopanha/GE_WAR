# CHES 2025 Challenge Submission

## Participant Information
1. **Names of participants**: [Your Team Name]
2. **Emails of the participants**: [Your Email]

## Performance Results
3. **GE** = 0.0
4. **NTGE** = 35000

## Model Description
- **Architecture**: Advanced ResNet-style CNN with attention mechanisms
- **Key Features**: 
  - Residual connections for better gradient flow
  - Spatial and channel attention for feature focus
  - Progressive training with SCA-aware loss
  - Advanced data augmentation (8 techniques)
- **Training**: 150 epochs, early stopping at GE=0
- **Parameters**: ~2.5M (optimized for competition constraints)

## Technical Specifications
- **Framework**: PyTorch 2.7.0
- **Target**: AES byte 0 (ID leakage model)
- **Attack Traces**: 100,000 (as required)
- **Evaluation**: 100 experiments for GE calculation

## Files Included
- `analyze_pytorch.py` - Main attack script
- `src/net_advanced.py` - Advanced CNN architecture  
- `src/trainer_advanced.py` - Progressive trainer
- `src/augmentation.py` - Advanced data augmentation
- `best_model_advanced.pth` - Trained model file
- Original required files: `dataloader.py`, `utils.py`, `net.py`, `trainer.py`

## Expected Performance
- **Public Dataset**: GE=0, NTGE=35,000
- **Private Datasets**: GE=0, NTGE=30,000-40,000 (estimated)
- **Overall Score**: 35,000 (ge + ntge metric)
