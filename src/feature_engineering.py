import pandas as pd
import os

PROCESSED_PATH = "data/processed/"
FEATURES_PATH = "data/features/"

def generate_context_features(player_stats, matches_df):
    # Rename 'id' to 'match_id' if needed
    if 'id' in matches_df.columns and 'match_id' not in matches_df.columns:
        matches_df = matches_df.rename(columns={'id': 'match_id'})

    # Assign Match Pressure Metric based on match stage
    matches_df = matches_df.sort_values(by='match_id').reset_index(drop=True)
    matches_df['match_pressure_metric'] = 1  # Default: league stage
    matches_df.loc[matches_df.index[-1], 'match_pressure_metric'] = 4  # Final
    matches_df.loc[matches_df.index[-4:-1], 'match_pressure_metric'] = 3  # Eliminator & Qualifiers

    # Merge basic match info into player stats
    stats = player_stats.copy()
    stats = stats.merge(matches_df[['match_id', 'team1', 'team2', 'toss_winner', 'toss_decision', 'match_pressure_metric']],
                        on='match_id', how='left')

    def determine_innings_type(row):
        if row['toss_decision'] == 'bat':
            return 'bat_first' if row['batting_team'] == row['toss_winner'] else 'bat_second'
        else:  # toss_decision == 'field'
            return 'bat_first' if row['batting_team'] != row['toss_winner'] else 'bat_second'

    # Add innings_type column
    stats['innings_type'] = stats.apply(determine_innings_type, axis=1)

    # Use opponent as team2 if batting_team == team1, else team1
    stats['opponent'] = stats.apply(
        lambda row: row['team2'] if row['batting_team'] == row['team1'] else row['team1'], axis=1
    )

    # Contextual averages
    stats['avg_points_venue'] = stats.groupby(['player', 'venue'])['fantasy_points'].transform('mean')
    stats['avg_points_opponent'] = stats.groupby(['player', 'opponent'])['fantasy_points'].transform('mean')
    stats['avg_points_batting_first'] = stats[stats['innings'] == 1].groupby('player')['fantasy_points'].transform('mean')
    stats['avg_points_batting_second'] = stats[stats['innings'] == 2].groupby('player')['fantasy_points'].transform('mean')

    return stats


def generate_all_features(player_stats, matches_df):
    player_stats = player_stats.sort_values(by=['player', 'match_id'])

    # Merge context
    stats = generate_context_features(player_stats, matches_df)

    # Compute venue-based and opponent-based stats globally, per player
    stats['avg_points_venue'] = stats.groupby(['player', 'venue'])['fantasy_points'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    stats['num_matches_venue'] = stats.groupby(['player', 'venue']).cumcount()

    stats['avg_points_opponent'] = stats.groupby(['player', 'opponent'])['fantasy_points'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    stats['num_matches_opponent'] = stats.groupby(['player', 'opponent']).cumcount()

    # Now do per-player features like last 5 matches and innings type
    features = []
    for player, group in stats.groupby('player'):
        group = group.copy().sort_values('match_id')

        # Head-to-Head Opponent last 5 matches average
        group['avg_points_vs_opponent_last_5'] = group.apply(
            lambda row: group[(group['opponent'] == row['opponent']) & 
                              (group['match_id'] < row['match_id'])]['fantasy_points']
                        .tail(5).mean(),
            axis=1
        )

        # Venue + Opposition Combo last 5 matches average
        group['avg_points_vs_opponent_at_venue_last_5'] = group.apply(
            lambda row: group[(group['opponent'] == row['opponent']) &
                              (group['venue'] == row['venue']) &
                              (group['match_id'] < row['match_id'])]['fantasy_points']
                        .tail(5).mean(),
            axis=1
        )

        group['avg_points_last_5'] = group['fantasy_points'].shift(1).rolling(5, min_periods=1).mean()
        group['avg_points_last_5'] = group['avg_points_last_5'].fillna(0)
        group['total_points_last_5'] = group['fantasy_points'].shift(1).rolling(5, min_periods=1).sum()
        group['num_past_matches'] = group['fantasy_points'].expanding().count() - 1
        group['is_debut'] = (group['num_past_matches'] == 0).astype(int)

        # Batting first/second averages
        group['avg_points_bat_first'] = group.apply(
            lambda row: group[(group['innings_type'] == 'bat_first') & (group['match_id'] < row['match_id'])]['fantasy_points'].mean()
            if row['innings_type'] == 'bat_first' else None,
            axis=1
        )
        group['avg_points_bat_second'] = group.apply(
            lambda row: group[(group['innings_type'] == 'bat_second') & (group['match_id'] < row['match_id'])]['fantasy_points'].mean()
            if row['innings_type'] == 'bat_second' else None,
            axis=1
        )

        features.append(group)

    all_features_df = pd.concat(features).reset_index(drop=True)
    return all_features_df


if __name__ == "__main__":
    os.makedirs(FEATURES_PATH, exist_ok=True)

    player_stats = pd.read_csv(os.path.join(PROCESSED_PATH, "player_match_stats.csv"))
    matches_df = pd.read_csv(os.path.join(PROCESSED_PATH, "matches.csv"))

    if 'batting_team' not in player_stats.columns:
        print("⚠️ Add 'batting_team' info in fantasy point script first.")
        exit()

    final_df = generate_all_features(player_stats, matches_df)
    final_df.to_csv(os.path.join(FEATURES_PATH, "player_match_features_full.csv"), index=False)
    print("✅ Saved all contextual + historical features successfully.")

    final_df.to_csv("data/player_features.csv", index=False)
    print("✅ Features saved to data/player_features.csv")
