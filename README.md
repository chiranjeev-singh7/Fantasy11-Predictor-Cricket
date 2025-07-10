# 🏏 Fantasy11 Predictor - Intelligent Dream11 Team Selector

This project is a complete pipeline for predicting the **optimal Dream11 fantasy cricket team** based on IPL data. It leverages advanced **feature engineering**, precise fantasy point calculations, and machine learning (Logistic Regression) to identify players most likely to perform well and constructs a fantasy XI while respecting team constraints.

---

## 📁 Project Structure

```text
Fantasy11-Predictor/
│
├── data/
│   ├── raw/                # Original IPL datasets (deliveries.csv, matches.csv)
│   ├── processed/          # Fantasy point calculations and match preprocessing
│   └── features/           # Contextual + historical features for each player
│
├── models/
│   └── model_dream11.pkl   # Trained logistic regression model
│
├── notebook/
│   └── model_dream11-2.ipynb   # Model training and evaluation notebook
│
├── src/
│   ├── data_loader.py
│   ├── fantasy_points.py
│   ├── feature_engineering.py
│   ├── save_matches.py
│   └── dream11_predictor.py
│
├── requirements.txt
└── README.md
```



---

## 🚀 How It Works

### 1. Data Loading

`data_loader.py` reads:
- `matches.csv` — overall match metadata
- `deliveries.csv` — ball-by-ball data for every IPL match

These datasets are merged to build a unified table with each delivery’s context.

---

### 2. Fantasy Point Calculation

`fantasy_points.py` computes official Dream11 points for every player’s match:
- Batting points (runs, boundaries, milestones)
- Bowling points (wickets, economy rates)
- Fielding points (catches, stumpings, run-outs)
- Penalties for ducks or slow strike rates
- Starter bonus for simply playing

Generates:  data/processed/player_match_stats.csv


---

### 3. Feature Engineering — ⭐️ The Heart of the Project

This is where your work shines!

The file: data/player_features.csv is a **highly detailed, feature-rich dataset** where each row represents a player’s performance context in a given match. It includes:

✅ **Venue-specific performance**
- How a player performs at a specific stadium.
- E.g. *“Virat Kohli’s record at Chinnaswamy.”*

✅ **Opponent-specific performance**
- A player’s historical average vs a particular team.
- E.g. *“Virat vs Rajasthan Royals overall.”*

✅ **Venue + Opponent combination**
- How a player performs against a specific team at a particular venue.
- E.g. *“Virat vs Rajasthan at Chinnaswamy.”*

✅ **Batting first vs second performance**
- Separate averages for batting first or chasing.
- E.g. *“Virat’s average when opening the innings vs chasing totals.”*

✅ **Recent form metrics**
- Rolling averages for:
  - Last 5 matches overall
  - Last 5 matches against a specific opponent
  - Last 5 matches against opponent at same venue
- E.g. *“Virat’s average fantasy points in last 5 games”* or *“Virat’s last 5 matches vs Rajasthan at Chinnaswamy.”*

✅ **Match context**
- Match pressure level:
  - League stage
  - Eliminators / Qualifiers
  - Final
- Toss winner and decision
- Innings type (bat first / bat second)

✅ **Player history**
- Number of past matches played
- Debut indicator flag

---

**Example features in player_features.csv:**

| player          | venue         | opponent          | avg_points_venue | avg_points_opponent | avg_points_bat_first | avg_points_last_5 | avg_points_vs_opponent_last_5 | ... |
|-----------------|---------------|-------------------|------------------|---------------------|----------------------|-------------------|-------------------------------|-----|
| Virat Kohli     | M Chinnaswamy | Rajasthan Royals  | 52.4             | 48.2                | 56.1                 | 50.3              | 47.5                          | ... |

This deep, **contextual understanding** is precisely what allows the model to make highly targeted predictions for future matches.

---

### 4. Model Training

Modeling is done in: notebook/model_dream11-2.ipynb

- Logistic Regression trained on whether a player scored above median fantasy points.
- Trained on the contextual features from `player_features.csv`.
- Saves trained model as: models/model_dream11.pkl

---

### 5. Dream11 Prediction

Run:
```bash
python src/dream11_predictor.py
```

---

### 6. Output

Choose either:

- Predict by match ID(refer to data/raw/matches.csv)
- Predict by year + team1 + team2 + encounter number

The predictor:

- Loads player features for the selected match
- Uses the trained model to estimate each player’s probability of being a top performer
- Selects the top 11 players, enforcing:
- No more than 7 players from one team

Assigns:

- (C) — Captain
- (VC) — Vice-captain

Example Output:

🎯 Predicted Dream11:

         player              team    dream11_prob
     V Kohli (C)       Bangalore           0.935
     F du Plessis (VC) Bangalore           0.912
     G Maxwell         Bangalore           0.902
     S Samson          Rajasthan           0.882
     Y Chahal          Rajasthan           0.860
     T Boult           Rajasthan           0.845
     R Ashwin          Rajasthan           0.832
     D Karthik         Bangalore           0.810
     M Siraj           Bangalore           0.802
     R Parag           Rajasthan           0.795
     H Patel           Bangalore           0.790
     

---

🔥 Why This Predictor Is Powerful

✅ Learns from player-specific patterns, not just overall averages
✅ Considers:

  - Venue-specific records
  - Opponent-specific records
  - Batting first vs chasing
  - Recent form
  - High-pressure matches

✅ Adheres to Dream11 constraints (max 7 players from one team)
✅ Automatically infers unknown player teams from deliveries data if missing
✅ Smart captain/vice-captain selection


---

📈 Model Details

- Algorithm: Logistic Regression
- Prediction type: Instead of simply classifying players as “Good” or “Not Good” (0 or 1), the model outputs a probability for each player. This probability tells us how confident the model is that the player will score above the median fantasy points in that match.
- In the predictor code, this happens via:
  ```
  match_players['dream11_prob'] = model.predict_proba(X)[:, 1]
  ```
This means we’re not just labeling players good/bad, but measuring how far their performance likelihood is from the decision boundary. Higher probabilities imply a higher chance of a strong fantasy performance, and this ranking is crucial for choosing the top 11 players.

Data used:

- Training data: IPL seasons 2008–2023
- Testing data: IPL 2024 matches

🧪 Model Evaluation (2024 test):

📊 RandomForest — F1 Score: 0.9056
📊 GradientBoosting — F1 Score: 0.9080
📊 LogisticRegression — F1 Score: 0.9100

Features:

  - Contextual averages
  - Rolling stats
  - Historical match performance

Evaluation metrics:

  - ROC-AUC
  - Log-loss
  - Cross-validation






