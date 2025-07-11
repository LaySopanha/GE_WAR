import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from src.net import MLP, CNN, weight_init
from torch.optim.lr_scheduler import StepLR # <-- Import the scheduler
from tqdm import tqdm

def trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device, writer=None,
            X_val=None, plt_val=None, leakage_fn=None, correct_key=None):

    # Build the model
    if model_type == "mlp":
        model = MLP(config, num_sample_pts, classes).to(device)
    elif model_type == "cnn":
        model = CNN(config, num_sample_pts, classes).to(device)
    weight_init(model, config['kernel_initializer'])

    # Creates the optimizer
    lr = config["lr"]
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # --- 1. SETUP SCHEDULER AND EARLY STOPPING ---
    # This scheduler will decrease the LR by a factor of 0.1 every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Early Stopping parameters
    patience = 15  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0
    best_val_rank = float('inf')
    # This dictionary will store the metrics from the single best epoch found
    final_metrics_best_epoch = {'val_loss': float('inf'), 'val_acc': 0.0, 'val_rank': float('inf')}
    # ---

    criterion = nn.CrossEntropyLoss()
    start = time.time()

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            tk0 = tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1}/{num_epochs} [{phase.capitalize()}]")
            for (traces, labels) in tk0:
                inputs = traces.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                tk0.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tqdm.write(f"--> Epoch {epoch+1} {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Log loss and accuracy to TensorBoard
            if writer:
                if phase == 'train':
                    writer.add_scalar('Loss/train', epoch_loss, global_step=epoch)
                    writer.add_scalar('Accuracy/train', epoch_acc, global_step=epoch)
                else:  # phase == 'val'
                    writer.add_scalar('Loss/val', epoch_loss, global_step=epoch)
                    writer.add_scalar('Accuracy/val', epoch_acc, global_step=epoch)

            # VALIDATION KEY RANK CALCULATION and EARLY STOPPING LOGIC
            if phase == 'val' and X_val is not None:
                # --- This block now calculates rank and decides whether to stop ---
                model.eval()
                # (You can add the batched evaluation here for memory safety if needed)
                with torch.no_grad():
                    val_inputs = torch.from_numpy(X_val).float().to(device).unsqueeze(1)
                    predictions = model(val_inputs)
                    probabilities = F.softmax(predictions, dim=1).cpu().numpy()

                all_scores = np.zeros(256)
                for k_guess in range(256):
                    hypothetical_labels = leakage_fn(plt_val, k_guess)
                    probs_for_key = probabilities[np.arange(len(hypothetical_labels)), hypothetical_labels]
                    all_scores[k_guess] = np.sum(np.log(probs_for_key + 1e-36))

                sorted_indices = np.argsort(all_scores)[::-1]
                validation_key_rank = np.where(sorted_indices == correct_key)[0][0]

                if writer:
                    writer.add_scalar('Rank/validation_key_rank', validation_key_rank, global_step=epoch)
                tqdm.write(f"--> Epoch {epoch+1} Validation Key Rank: {validation_key_rank}")

                # --- 2. IMPLEMENT EARLY STOPPING LOGIC ---
                if validation_key_rank < best_val_rank:
                    best_val_rank = validation_key_rank
                    epochs_no_improve = 0
                    # Store the metrics from this new best epoch
                    final_metrics_best_epoch['val_loss'] = epoch_loss
                    final_metrics_best_epoch['val_acc'] = epoch_acc.item() # Use .item() to get scalar
                    final_metrics_best_epoch['val_rank'] = float(validation_key_rank)
                    # You could also save the best model's weights here
                    # torch.save(model.state_dict(), "best_model_so_far.pth")
                    tqdm.write(f"*** New best rank found: {best_val_rank} at epoch {epoch + 1} ***")
                else:
                    epochs_no_improve += 1

        # --- 3. STEP THE SCHEDULER AND CHECK FOR EARLY STOPPING ---
        scheduler.step()
        if writer:
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step=epoch)

        if epochs_no_improve >= patience:
            print(f"\n--- Early stopping triggered at epoch {epoch + 1}. Best rank was {best_val_rank}. ---")
            break  # Exit the main training loop

    time_elapsed = time.time() - start
    print(f"Finished Training Model in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Return both the trained model and the metrics from the BEST epoch found
    return model, final_metrics_best_epoch