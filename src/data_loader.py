import pandas as pd
import os

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def load_data():
    matches = pd.read_csv(os.path.join(RAW_PATH, "matches.csv"))
    deliveries = pd.read_csv(os.path.join(RAW_PATH, "deliveries.csv"))

    # Fix capital 'Season'
    matches.rename(columns={'Season': 'season'}, inplace=True)

    # Select only relevant columns to merge
    merged = deliveries.merge(
        matches[['id', 'season', 'date', 'venue', 'team1', 'team2']],
        left_on='match_id', right_on='id', how='left'
    )

    return matches, merged

if __name__ == "__main__":
    matches, merged = load_data()
    print("Matches shape:", matches.shape)
    print("Merged shape:", merged.shape)
