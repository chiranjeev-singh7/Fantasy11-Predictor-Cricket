import pandas as pd
import numpy as np
import joblib

# === Load resources ===
model = joblib.load("models/model_dream11.pkl")
features_df = pd.read_csv("data/player_features.csv")
matches_df = pd.read_csv("data/raw/matches.csv")
deliveries_df = pd.read_csv("data/raw/deliveries.csv")

# Normalize season column
matches_df['season'] = matches_df['season'].astype(str).str.extract(r'(\d{4})').astype(int)

# === Helper to infer team from deliveries ===
def infer_team_from_deliveries(player, match_id):
    deliveries_match = deliveries_df[deliveries_df['match_id'] == match_id]

    # First check for batting involvement
    batting_involvement = deliveries_match[
        (deliveries_match['batter'] == player) |
        (deliveries_match['non_striker'] == player)
    ]
    if not batting_involvement.empty:
        team = batting_involvement['batting_team'].mode()
        if not team.empty:
            return team.iloc[0]

    # Then check for bowling involvement
    bowling_involvement = deliveries_match[deliveries_match['bowler'] == player]
    if not bowling_involvement.empty:
        team = bowling_involvement['bowling_team'].mode()
        if not team.empty:
            return team.iloc[0]

    # If nothing found
    return 'Unknown'


# === Match ID finder ===
def get_match_id_from_details(year, team1, team2, encounter_no):
    team1, team2 = team1.strip(), team2.strip()
    df_filtered = matches_df[
        (matches_df['season'] == year) &
        (((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
         ((matches_df['team1'] == team2) & (matches_df['team2'] == team1)))
    ].sort_values(by='id').reset_index(drop=True)

    if encounter_no > len(df_filtered) or encounter_no < 1:
        raise ValueError(f"❌ Encounter {encounter_no} not found for {team1} vs {team2} in {year}.")
    
    return df_filtered.iloc[encounter_no - 1]['id'], len(df_filtered)

# === Dream11 Predictor ===
def predict_dream11(match_id):
    match_players = features_df[features_df['match_id'] == match_id].copy()

    # Fix team assignment from batting_team and bowling_team
    match_players['team'] = match_players['batting_team'].astype(str)
    match_players['team'] = match_players['team'].replace(['0', 'nan', '', 'None'], pd.NA)
    match_players['team'] = match_players['team'].fillna(match_players['bowling_team'].astype(str))
    match_players['team'] = match_players['team'].replace(['0', 'nan', '', 'None'], 'Unknown')

    # Infer team for Unknown players using deliveries.csv
    match_players['team'] = match_players.apply(
        lambda row: infer_team_from_deliveries(row['player'], row['match_id']) if row['team'] == 'Unknown' else row['team'],
        axis=1
    )

    print(f"\n✅ Found {len(match_players)} players for match ID {match_id}.")

    feature_columns = [
        'innings', 'runs', 'balls_faced', 'fours', 'sixes', 'duck', 'half', 'century',
        'balls', 'conceded', 'wickets', 'maidens', 'caught', 'run out', 'stumped',
        'match_pressure_metric', 'avg_points_venue', 'avg_points_opponent',
        'avg_points_batting_first', 'avg_points_batting_second',
        'num_matches_venue', 'num_matches_opponent',
        'avg_points_vs_opponent_last_5', 'avg_points_vs_opponent_at_venue_last_5',
        'avg_points_last_5', 'total_points_last_5', 'num_past_matches', 'is_debut',
        'avg_points_bat_first', 'avg_points_bat_second'
    ]
    
    for col in feature_columns:
        if col not in match_players.columns:
            match_players[col] = 0

    X = match_players[feature_columns].fillna(0)
    match_players['dream11_prob'] = model.predict_proba(X)[:, 1]

    top11 = match_players.sort_values(by='dream11_prob', ascending=False).head(11).reset_index(drop=True)

    # Cap max 7 players per team
    team_count = top11['team'].value_counts()
    if any(team_count > 7):
        excess_team = team_count.idxmax()
        excess_count = team_count[excess_team] - 7
        keep = top11[top11['team'] != excess_team]
        drop = top11[top11['team'] == excess_team].sort_values(by='dream11_prob', ascending=False).iloc[:-excess_count]
        others = match_players[~match_players['player'].isin(top11['player']) & (match_players['team'] != excess_team)]
        replace = others.sort_values(by='dream11_prob', ascending=False).head(excess_count)
        top11 = pd.concat([drop, keep, replace]).sort_values(by='dream11_prob', ascending=False).head(11).reset_index(drop=True)

    if len(top11) >= 2:
        top11.at[0, 'player'] += " (C)"
        top11.at[1, 'player'] += " (VC)"
    elif len(top11) == 1:
        top11.at[0, 'player'] += " (C)"

    print("\n🎯 Predicted Dream11:\n")
    print(top11[['player', 'team', 'dream11_prob']].to_string(index=False))
    return top11[['player', 'team', 'dream11_prob']]

# === Entry Point ===
if __name__ == "__main__":
    mode = input("🔍 Enter 1 to predict by match ID or 2 to predict by year + teams + encounter number: ")

    if mode == "1":
        match_id_input = int(input("🔢 Enter match ID: "))
        predict_dream11(match_id_input)

    elif mode == "2":
        year_input = int(input("📅 Enter year (e.g., 2023): "))
        teams_played = pd.unique(matches_df[matches_df['season'] == year_input][['team1', 'team2']].values.ravel())
        teams_played = sorted([t for t in teams_played if isinstance(t, str) and t.strip() != ''])
        print("\n📋 Teams that played in", year_input, ":\n", ", ".join(teams_played))

        team1_input = input("\n🔹 Enter Team 1 name from above: ").strip()
        team2_input = input("🔸 Enter Team 2 name from above: ").strip()

        try:
            match_id, total_encounters = get_match_id_from_details(year_input, team1_input, team2_input, 1)
            print(f"\n📊 {team1_input} and {team2_input} faced each other {total_encounters} times in {year_input}.")
            encounter_no_input = int(input("🔁 Enter encounter number (e.g., 1 for their first match): "))
            match_id_final, _ = get_match_id_from_details(year_input, team1_input, team2_input, encounter_no_input)
            predict_dream11(match_id_final)
        except ValueError as ve:
            print(ve)
    else:
        print("❌ Invalid input. Please enter 1 or 2.")
