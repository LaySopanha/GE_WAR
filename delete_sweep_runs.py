import wandb

# --- Configuration ---
ENTITY = "panhalay69420-idri"
PROJECT = "GE_WAR"
SWEEP_ID = "y8ev04fd"
# -------------------

api = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

for run in sweep.runs:
    print(f"Deleting run: {run.name} ({run.id})")
    run.delete()

print("\nAll runs in the sweep have been deleted.")
