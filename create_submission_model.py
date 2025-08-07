#!/usr/bin/env python3
"""
Model distillation: Train a single model that captures ensemble knowledge
"""
import os
import torch
import torch.nn as nn
import numpy as np
from src.net import CNN
from src.dataloader import Custom_Dataset
from src.utils import AES_Sbox
import glob
import json

def create_distilled_model(teacher_models, X_train, Y_train, device, config):
    """Train a single student model using ensemble teacher knowledge"""
    
    # Create student model (same architecture as teachers)
    poi_width = len(X_train[0])
    classes = 256 if config['leakage'] == 'ID' else 9
    
    search_space = {
        'conv_layers': 2,
        'filters': 16,
        'kernels': 36,
        'dropout_rate': 0.27,  # Competition-tested value
        'activation': 'selu'
    }
    
    student_model = CNN(search_space, poi_width, classes).to(device)
    
    # Get soft targets from teacher ensemble
    teacher_predictions = []
    
    print(f"📚 Generating soft targets from {len(teacher_models)} teachers...")
    
    for teacher_info in teacher_models:
        teacher = CNN(search_space, poi_width, classes).to(device)
        teacher.load_state_dict(torch.load(teacher_info['model_file'], map_location=device))
        teacher.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_train).to(device)
            teacher_pred = teacher(X_tensor).cpu().numpy()
            teacher_predictions.append(teacher_pred)
    
    # Average teacher predictions (soft targets)
    soft_targets = np.mean(teacher_predictions, axis=0)
    
    # Train student with knowledge distillation
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0005)
    criterion_hard = nn.CrossEntropyLoss()
    criterion_soft = nn.KLDivLoss(reduction='batchloss')
    
    student_model.train()
    
    print("🎓 Training distilled model...")
    
    for epoch in range(50):  # Shorter training since we have soft targets
        total_loss = 0
        
        for i in range(0, len(X_train), 64):  # Batch size 64
            batch_X = torch.FloatTensor(X_train[i:i+64]).to(device)
            batch_Y_hard = torch.LongTensor(Y_train[i:i+64]).to(device)
            batch_Y_soft = torch.FloatTensor(soft_targets[i:i+64]).to(device)
            
            optimizer.zero_grad()
            
            student_pred = student_model(batch_X)
            
            # Combined loss: hard targets + soft targets
            loss_hard = criterion_hard(student_pred, batch_Y_hard)
            loss_soft = criterion_soft(torch.log_softmax(student_pred, dim=1), 
                                     torch.softmax(batch_Y_soft, dim=1))
            
            # Weight the losses (more emphasis on soft targets for generalization)
            total_loss_batch = 0.3 * loss_hard + 0.7 * loss_soft
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    
    return student_model

def create_competition_submission():
    """Create the best single model for competition submission"""
    
    # Load all robust models
    model_files = glob.glob("ge0_robust_model_run_*.pth")
    
    if len(model_files) < 2:
        print("❌ Need at least 2 robust models for distillation")
        return
    
    # Load training data
    dataset_obj = Custom_Dataset(root='../', dataset='CHES_2025', leakage='ID',
                                poi_start=0, poi_end=7000, train_end=500000, test_end=100000)
    
    # Use saved POI selection
    poi_metadata = np.load('poi_selection_metadata.npy', allow_pickle=True).item()
    top_k_indices = np.array(poi_metadata['poi_indices'])
    
    X_train = dataset_obj.X_profiling[:, top_k_indices]
    Y_train = dataset_obj.Y_profiling
    
    # Load teacher models info
    teachers = []
    for model_file in model_files:
        base_name = model_file.replace("ge0_robust_model_", "").replace(".pth", "")
        metadata_file = f"ge0_robust_metadata_{base_name}.json"
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            teachers.append({
                'model_file': model_file,
                'metadata': metadata
            })
    
    # Sort by robustness and select top teachers
    teachers.sort(key=lambda x: x['metadata'].get('robustness_penalty', 999))
    top_teachers = teachers[:3]  # Use top 3 most robust
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create distilled model
    config = {'leakage': 'ID'}
    distilled_model = create_distilled_model(top_teachers, X_train, Y_train, device, config)
    
    # Save the distilled model for submission
    torch.save(distilled_model.state_dict(), "competition_submission_distilled.pth")
    
    # Save metadata
    submission_metadata = {
        'model_type': 'distilled_ensemble',
        'num_teachers': len(top_teachers),
        'teacher_models': [t['model_file'] for t in top_teachers],
        'poi_indices': top_k_indices.tolist(),
        'num_poi': len(top_k_indices),
        'leakage': 'ID',
        'dataset': 'CHES_2025'
    }
    
    np.save("competition_submission_metadata.npy", submission_metadata)
    
    print(f"✅ Created competition submission: competition_submission_distilled.pth")
    print(f"📊 Distilled knowledge from {len(top_teachers)} robust models")
    
    return distilled_model

if __name__ == "__main__":
    create_competition_submission()
