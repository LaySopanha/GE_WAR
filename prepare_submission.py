#!/usr/bin/env python3
"""
CHES 2025 Submission Preparation Script
Prepares the submission directory according to competition rules
"""

import os
import shutil
import zipfile
from pathlib import Path

def prepare_submission(team_name="YourTeam", submission_no=0):
    """Prepare CHES 2025 submission directory"""
    
    # Submission directory name
    submission_dir = f"ches2025_pytorch_{team_name}_{submission_no}"
    
    print(f"🚀 Preparing CHES 2025 Submission: {submission_dir}")
    
    # Create submission directory
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    # Required files to copy
    required_files = [
        "analyze_pytorch.py",
        "submission.md"
    ]
    
    # Required src files
    src_files = [
        "src/dataloader.py",
        "src/utils.py", 
        "src/net.py",
        "src/trainer.py"
    ]
    
    # Additional advanced files (allowed but not required)
    additional_files = [
        "src/net_advanced.py",
        "src/trainer_advanced.py", 
        "src/augmentation.py"
    ]
    
    # Model files to check
    model_files = [
        "best_model_advanced.pth",
        "outputs/best_model_advanced.pth",
        "best_model.pth"
    ]
    
    print("\n📁 Copying required files...")
    
    # Copy required files
    for file in required_files:
        if os.path.exists(file):
            shutil.copy2(file, submission_dir)
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND!")
            return False
    
    # Create src directory
    src_dir = os.path.join(submission_dir, "src")
    os.makedirs(src_dir)
    
    # Copy required src files
    for file in src_files:
        if os.path.exists(file):
            shutil.copy2(file, src_dir)
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND!")
            return False
    
    # Copy additional files if they exist
    print("\n📦 Copying additional advanced files...")
    for file in additional_files:
        if os.path.exists(file):
            shutil.copy2(file, src_dir)
            print(f"  ✅ {file} (advanced)")
        else:
            print(f"  ⚠️ {file} - not found (optional)")
    
    # Copy model file
    print("\n🤖 Copying model file...")
    model_copied = False
    for model_file in model_files:
        if os.path.exists(model_file):
            # Copy to submission root with standard name
            if "advanced" in model_file:
                dest_name = "best_model_advanced.pth"
            else:
                dest_name = "best_model.pth"
                
            shutil.copy2(model_file, os.path.join(submission_dir, dest_name))
            print(f"  ✅ {model_file} -> {dest_name}")
            model_copied = True
            break
    
    if not model_copied:
        print("  ❌ No trained model found! Train a model first.")
        return False
    
    # Validate submission
    print("\n🔍 Validating submission...")
    
    required_structure = [
        "analyze_pytorch.py",
        "submission.md",
        "src/dataloader.py",
        "src/utils.py",
        "src/net.py", 
        "src/trainer.py"
    ]
    
    all_valid = True
    for item in required_structure:
        path = os.path.join(submission_dir, item)
        if os.path.exists(path):
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} - MISSING!")
            all_valid = False
    
    # Check for model file
    model_files_in_submission = [f for f in os.listdir(submission_dir) if f.endswith('.pth')]
    if model_files_in_submission:
        print(f"  ✅ Model file: {model_files_in_submission[0]}")
    else:
        print("  ❌ No .pth model file found!")
        all_valid = False
    
    if not all_valid:
        print("\n❌ Submission validation FAILED!")
        return False
    
    # Create zip file
    print(f"\n📦 Creating zip file: {submission_dir}.zip")
    with zipfile.ZipFile(f"{submission_dir}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, os.path.dirname(submission_dir))
                zipf.write(file_path, arc_path)
    
    # Final summary
    print(f"\n🎉 Submission prepared successfully!")
    print(f"📂 Directory: {submission_dir}/")
    print(f"📦 Zip file: {submission_dir}.zip")
    print(f"📏 Zip size: {os.path.getsize(f'{submission_dir}.zip') / 1024 / 1024:.1f} MB")
    
    print(f"\n📋 Submission contents:")
    for root, dirs, files in os.walk(submission_dir):
        level = root.replace(submission_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            size_kb = os.path.getsize(os.path.join(root, file)) / 1024
            print(f"{subindent}{file} ({size_kb:.1f} KB)")
    
    print(f"\n✅ Ready for submission to CHES 2025!")
    print(f"🚀 Expected performance: GE=0, NTGE=35,000-50,000")
    
    return True

def check_compliance():
    """Check if submission meets all competition requirements"""
    print("🔍 Checking CHES 2025 compliance...")
    
    checks = []
    
    # Check PyTorch version
    try:
        import torch
        version = torch.__version__
        if version.startswith('2.7'):
            checks.append(("PyTorch 2.7.x", True, f"Found {version}"))
        else:
            checks.append(("PyTorch 2.7.x", False, f"Found {version} (wrong version)"))
    except ImportError:
        checks.append(("PyTorch 2.7.x", False, "PyTorch not installed"))
    
    # Check required files
    required_files = [
        "analyze_pytorch.py",
        "src/dataloader.py", 
        "src/utils.py",
        "src/net.py",
        "src/trainer.py"
    ]
    
    for file in required_files:
        exists = os.path.exists(file)
        checks.append((f"Required file: {file}", exists, "Found" if exists else "Missing"))
    
    # Check for trained model
    model_files = ["best_model_advanced.pth", "outputs/best_model_advanced.pth", "best_model.pth"]
    model_found = any(os.path.exists(f) for f in model_files)
    checks.append(("Trained model (.pth)", model_found, "Found" if model_found else "Missing"))
    
    # Print results
    print("\n📋 Compliance Check Results:")
    all_passed = True
    for check_name, passed, details in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {details}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All compliance checks PASSED!")
        print("🚀 Ready to prepare submission!")
    else:
        print("\n⚠️ Some compliance checks FAILED!")
        print("🔧 Fix the issues above before submitting.")
    
    return all_passed

if __name__ == "__main__":
    print("🏆 CHES 2025 Submission Preparation Tool")
    print("=" * 50)
    
    # Check compliance first
    if check_compliance():
        print("\n" + "=" * 50)
        
        # Get team info
        team_name = input("Enter your team name (default: YourTeam): ").strip()
        if not team_name:
            team_name = "YourTeam"
        
        submission_no = input("Enter submission number (default: 0): ").strip()
        if not submission_no.isdigit():
            submission_no = 0
        else:
            submission_no = int(submission_no)
        
        # Prepare submission
        success = prepare_submission(team_name, submission_no)
        
        if success:
            print("\n🎯 Next steps:")
            print("1. Review the submission directory")
            print("2. Test with: python analyze_pytorch.py")
            print("3. Submit the .zip file to organizers")
            print("4. Wait for acknowledgment (within 2 working days)")
        else:
            print("\n❌ Submission preparation failed!")
            print("🔧 Fix the issues and try again.")
    else:
        print("\n🔧 Fix compliance issues first!")
