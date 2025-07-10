import pandas as pd
import os

PROCESSED_PATH = "data/processed/"

def compute_fantasy_points(df, matches_df):
    # Compute balls faced properly based on extras
    df['balls_faced'] = df['extras_type'].apply(lambda x: 0 if x in ['wides', 'noballs'] else 1)

    df['innings'] = df['inning']  # for consistency

    # Starting XI (simplified assumption)
    start = matches_df[['id', 'team1', 'team2']]
    df = df.merge(start, left_on='match_id', right_on='id', how='left')
    df['starting_points'] = df.groupby(['match_id', 'batter'])['match_id'].transform(lambda x: 4)

    # Batting stats
    batting = df.groupby(['match_id', 'batter', 'batting_team', 'bowling_team', 'venue', 'innings']).agg(
        runs=('batsman_runs', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        fours=('batsman_runs', lambda x: (x == 4).sum()),
        sixes=('batsman_runs', lambda x: (x == 6).sum())
    ).reset_index()

    batting['duck'] = ((batting['runs'] == 0) & (batting['balls_faced'] > 0)).astype(int)
    batting['half'] = ((batting['runs'] >= 50) & (batting['runs'] < 100)).astype(int)
    batting['century'] = (batting['runs'] >= 100).astype(int)

    batting.rename(columns={'batter': 'player'}, inplace=True)

    # Bowling stats
    bowl = df.groupby(['match_id', 'bowler']).agg({
        'ball': 'count',
        'total_runs': 'sum',
        'player_dismissed': lambda x: x.notna().sum(),
    }).reset_index()
    bowl.columns = ['match_id', 'player', 'balls', 'conceded', 'wickets']
    bowl['maidens'] = 0  # optional

    # Fielding stats
    f = df[df['dismissal_kind'].isin(['caught', 'run out', 'stumped'])]
    runouts = f[f['dismissal_kind'] == 'run out'].copy()
    runouts['thrower'] = runouts['fielder'].fillna("").apply(lambda x: x.split('/')[0].strip())

    field = f.groupby(['match_id', 'fielder', 'dismissal_kind']).size().unstack(fill_value=0).reset_index()
    field.columns.name = None
    field = field.rename(columns={'fielder': 'player'})
    for col in ['caught', 'run out', 'stumped']:
        if col not in field.columns:
            field[col] = 0

    # Merge all
    ps = batting.merge(bowl, on=['match_id', 'player'], how='outer').merge(field, on=['match_id', 'player'], how='outer')
    ps.fillna(0, inplace=True)

    # Points calculation
    def pt(row):
        p = 0
        p += 4  # starting
        p += row.runs
        p += row.fours * 1 + row.sixes * 2
        p += row.half * 8 + row.century * 16
        p -= row.duck * 2
        p += row.wickets * 25
        if row.wickets >= 5: p += 16
        elif row.wickets >= 4: p += 8
        p += row.get('caught', 0) * 8 + row.get('stumped', 0) * 12 + row.get('run out', 0) * 12

        # Economy rate bonus/penalty
        if row.balls >= 12:
            rpo = row.conceded / (row.balls / 6)
            if rpo < 4: p += 6
            elif rpo < 5: p += 4
            elif rpo < 6: p += 2
            elif rpo > 11: p -= 6
            elif rpo > 10: p -= 4
            elif rpo > 9: p -= 2

        # Batting SR penalty
        if row.balls_faced >= 10:
            sr = row.runs / (row.balls_faced / 100)
            if sr < 50: p -= 6
            elif sr < 60: p -= 4
            elif sr < 70: p -= 2

        return p

    ps['fantasy_points'] = ps.apply(pt, axis=1)

    return ps

if __name__ == "__main__":
    from data_loader import load_data
    matches, merged = load_data()
    player_stats = compute_fantasy_points(merged, matches)
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    player_stats.to_csv(os.path.join(PROCESSED_PATH, "player_match_stats.csv"), index=False)
    print("✅ Saved stats with official scoring + batting_team info!")
