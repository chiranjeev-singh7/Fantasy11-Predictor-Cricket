for i, row in match_players.iterrows():
    #     if row['team'] == 'Unknown':
    #         if team_counts.get(team1, 0) < 11:
    #             match_players.at[i, 'team'] = team1
    #             team_counts[team1] = team_counts.get(team1, 0) + 1
    #         elif team_counts.get(team2, 0) < 11:
    #             match_players.at[i, 'team'] = team2
    #             team_counts[team2] = team_counts.get(team2, 0) + 1
    #         else:
    #             print("⚠️ More than 11 players already assigned per team. Check data.")

    # if 'Unknown' in match_players['team'].values:
    #     print("⚠️ Some players still have unknown team. Please inspect.")
    #     print(match_players[match_players['team'] == 'Unknown'][['player', 'batting_team', 'bowling_team']])
    #     return []