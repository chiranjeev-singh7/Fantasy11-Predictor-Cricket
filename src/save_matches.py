import pandas as pd
import os

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Read original IPL matches dataset
matches = pd.read_csv(os.path.join(RAW_PATH, "matches.csv"))

# Select only needed columns
matches = matches[['id', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision']]

# Save processed
matches.to_csv(os.path.join(PROCESSED_PATH, "matches.csv"), index=False)
print("✅ Saved matches.csv to data/processed/")
