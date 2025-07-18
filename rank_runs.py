import wandb
import pandas as pd

# --- Configuration ---
ENTITY = "panhalay69420-idri"
PROJECT = "GE_WAR"
# ---------------------

api = wandb.Api()

# Get all runs from the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

# Create a list to hold the run data
run_data = []

for run in runs:
    # Get the summary metrics for the run
    summary = run.summary
    
    # Extract the metrics you want to rank by
    final_ge = summary.get("final_GE_at_max_traces")
    ntge = summary.get("final_NTGE")
    
    # Add the data to the list
    if final_ge is not None and ntge is not None:
        run_data.append({
            "Run ID": run.id,
            "Final GE": final_ge,
            "NTGE": ntge,
            **run.config
        })

# Create a pandas DataFrame from the run data
df = pd.DataFrame(run_data)

# Convert NTGE to numeric, coercing errors to NaN
df['NTGE'] = pd.to_numeric(df['NTGE'], errors='coerce')

# Replace NaN with a large number for sorting purposes
df['NTGE'].fillna(99999, inplace=True)


# Sort the DataFrame by the desired metrics
df_sorted_ge = df.sort_values(by="Final GE", ascending=True).head(10)
df_sorted_ntge = df.sort_values(by="NTGE", ascending=True).head(10)

# Print the ranked tables
print("--- Top 10 Runs by Final GE ---")
print(df_sorted_ge)
print("\n--- Top 10 Runs by NTGE ---")
print(df_sorted_ntge)
