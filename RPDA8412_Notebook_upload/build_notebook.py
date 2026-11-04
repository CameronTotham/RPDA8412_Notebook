"""Builds RPDA8412_Market_Efficiency.ipynb from the cells below (run: python build_notebook.py)."""
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Testing Market Efficiency in European Soccer Betting Markets Using Machine Learning
**RPDA8412 Research Project | Cameron Joshua Totham (ST10301625) | Supervisor: Dr Janine Upton**

This notebook implements the methodology in Chapter 3 of the Research Report:

1. Load the Kaggle European Soccer Database (Mathien, 2016) and keep the five target leagues.
2. Convert bookmaker odds to overround-free implied probabilities and build the consensus benchmark.
3. Engineer pre-match features (rolling form, Elo, squad strength, head-to-head) using only information dated before kick-off.
4. Split the matches chronologically (70% train, 15% validation, 15% test).
5. Tune, calibrate and compare Logistic Regression, Random Forest and XGBoost on probability quality (RPS, Log-Loss).
6. Run the overround-adjusted value-betting simulation and the permutation significance test.

Bookmaker odds are never used as model inputs. Every number reported in the Research Report comes from the outputs of this notebook.
""")

md("## 0. Setup")
code("""
import os, sqlite3, json, platform, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, scipy, xgboost
from IPython.display import display, Markdown
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
TIMER = {}
_t = time.perf_counter()
def lap(stage):
    # record the wall-clock seconds spent since the previous lap
    global _t
    now = time.perf_counter(); TIMER[stage] = TIMER.get(stage, 0.0) + (now - _t); _t = now
np.random.seed(SEED)
pd.set_option("display.max_columns", 50)

# Data location: a local copy of database.sqlite, or download through kagglehub when running in Colab.
DATA_DIR = Path(os.environ.get("SOCCER_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "database.sqlite"
if not DB_PATH.exists():
    import kagglehub  # available in Google Colab
    DB_PATH = Path(kagglehub.dataset_download("hugomathien/soccer")) / "database.sqlite"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

versions = pd.DataFrame({"package": ["python", "numpy", "pandas", "scikit-learn", "scipy", "xgboost"],
                         "version": [platform.python_version(), np.__version__, pd.__version__,
                                     sklearn.__version__, scipy.__version__, xgboost.__version__]})
versions
""")

md("## 1. Load the data and keep the five target leagues")
code("""
con = sqlite3.connect(DB_PATH)
leagues = pd.read_sql("SELECT l.id AS league_id, l.name AS league, c.name AS country FROM League l JOIN Country c ON l.country_id = c.id", con)
leagues
""")
code("""
TARGET = {"England": "Premier League", "Spain": "La Liga", "Germany": "Bundesliga", "Italy": "Serie A", "France": "Ligue 1"}
target_ids = leagues.loc[leagues["country"].isin(TARGET), ["league_id", "country"]]
assert len(target_ids) == 5, target_ids

match = pd.read_sql("SELECT * FROM Match", con)
match = match[match["league_id"].isin(target_ids["league_id"])].copy()
match["league"] = match["league_id"].map(dict(zip(target_ids["league_id"], target_ids["country"].map(TARGET))))
match["date"] = pd.to_datetime(match["date"])
match = match.sort_values(["date", "id"]).reset_index(drop=True)
teams = pd.read_sql("SELECT team_api_id, team_long_name FROM Team", con)

counts = match.pivot_table(index="season", columns="league", values="id", aggfunc="count", margins=True, margins_name="Total")
display(Markdown(f"**Matches in the five leagues before cleaning: {len(match):,}**"))
counts
""")

code("lap('1. load data')")
md("## 2. Bookmaker odds, implied probabilities and the consensus benchmark")
code("""
BOOKS = sorted({c[:-1] for c in match.columns if c.endswith("H") and c[:-1] + "D" in match.columns
                and c[:-1] + "A" in match.columns and c not in ("home_team_goal",)})
odds_cols = {b: [b + "H", b + "D", b + "A"] for b in BOOKS}
coverage = pd.Series({b: match[cols].notna().all(axis=1).mean() for b, cols in odds_cols.items()}, name="share of matches quoted")
coverage.to_frame().style.format("{:.1%}")
""")
code("""
warnings.filterwarnings("ignore", message="Mean of empty slice")   # the 8 matches with no odds at all
probs, overrounds, valid = [], [], []
for b, cols in odds_cols.items():
    o = match[cols].to_numpy(dtype=float)
    ok = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    inv = np.where(ok[:, None], 1.0 / o, np.nan)
    s = inv.sum(axis=1, keepdims=True)
    probs.append(inv / s)                # basic normalisation removes the overround
    overrounds.append(s[:, 0] - 1.0)
    valid.append(ok)
probs = np.stack(probs)                  # bookmakers x matches x 3
valid = np.stack(valid)
odds_arr = np.stack([match[c].to_numpy(dtype=float) for c in odds_cols.values()])
odds_arr = np.where(valid[:, :, None], odds_arr, np.nan)

match["n_books"] = valid.sum(axis=0)
for k, lab in enumerate(["H", "D", "A"]):
    match[f"p_mkt_{lab}"] = np.nanmean(probs[:, :, k], axis=0)
    match[f"odds_avg_{lab}"] = np.nanmean(odds_arr[:, :, k], axis=0)
    match[f"odds_max_{lab}"] = np.nanmax(np.where(np.isnan(odds_arr[:, :, k]), -np.inf, odds_arr[:, :, k]), axis=0)
match["overround_mean"] = np.nanmean(np.where(valid, np.stack(overrounds), np.nan), axis=0)

MIN_BOOKS = 3
dropped = match[match["n_books"] < MIN_BOOKS]
display(Markdown(f"Matches with fewer than {MIN_BOOKS} bookmakers quoting all three outcomes: **{len(dropped):,}** "
                 f"({len(dropped) / len(match):.2%}); of these, **{(dropped['n_books'] == 0).sum():,}** had no odds at all."))
match = match[match["n_books"] >= MIN_BOOKS].reset_index(drop=True)
match["outcome"] = np.select([match["home_team_goal"] > match["away_team_goal"],
                              match["home_team_goal"] == match["away_team_goal"]], [0, 1], 2)   # 0=H, 1=D, 2=A
summary = match.groupby("league").agg(matches=("id", "count"), books_mean=("n_books", "mean"),
                                      overround_mean=("overround_mean", "mean"),
                                      home=("outcome", lambda s: (s == 0).mean()),
                                      draw=("outcome", lambda s: (s == 1).mean()),
                                      away=("outcome", lambda s: (s == 2).mean()))
summary.loc["All"] = [len(match), match["n_books"].mean(), match["overround_mean"].mean(),
                      (match["outcome"] == 0).mean(), (match["outcome"] == 1).mean(), (match["outcome"] == 2).mean()]
summary.to_csv(OUT / "table_data_summary.csv")
summary.style.format({"matches": "{:,.0f}", "books_mean": "{:.2f}", "overround_mean": "{:.2%}",
                      "home": "{:.1%}", "draw": "{:.1%}", "away": "{:.1%}"})
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(match["overround_mean"] * 100, bins=40, color="#2E75B6", edgecolor="white")
ax.set_xlabel("Mean bookmaker overround per match (%)"); ax.set_ylabel("Matches")
ax.set_title("Distribution of the overround")
fig.tight_layout(); fig.savefig(OUT / "fig_overround.png", dpi=200)
""")

code("lap('2. odds and benchmark')")
md("""
## 3. Feature engineering
All features use only matches played before the match being predicted. Parameters are fixed in advance:
rolling windows of 5 and 10 matches, Elo initial rating 1,500 and K-factor 20.
""")
code("""
# ---- Elo ratings (pre-match), updated chronologically across all five leagues
ELO_START, ELO_K = 1500.0, 20.0
elo = {}
elo_home, elo_away = np.empty(len(match)), np.empty(len(match))
for i, (h, a, hg, ag) in enumerate(match[["home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"]].itertuples(index=False)):
    rh, ra = elo.get(h, ELO_START), elo.get(a, ELO_START)
    elo_home[i], elo_away[i] = rh, ra
    exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
    s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    elo[h] = rh + ELO_K * (s_h - exp_h)
    elo[a] = ra + ELO_K * ((1 - s_h) - (1 - exp_h))
match["elo_home"], match["elo_away"] = elo_home, elo_away
match["elo_diff"] = match["elo_home"] - match["elo_away"]
""")
code("""
# ---- Rolling form: home team over its previous home matches, away team over its previous away matches
def side_frame(side):
    opp = "away" if side == "home" else "home"
    f = pd.DataFrame({"id": match["id"], "date": match["date"], "team": match[f"{side}_team_api_id"],
                      "gf": match[f"{side}_team_goal"], "ga": match[f"{opp}_team_goal"]})
    f["pts"] = np.select([f["gf"] > f["ga"], f["gf"] == f["ga"]], [3, 1], 0)
    return f

for side in ("home", "away"):
    f = side_frame(side).sort_values(["team", "date", "id"])
    g = f.groupby("team")
    for w in (5, 10):
        for col in ("gf", "ga", "pts"):
            f[f"{side}_{col}_{w}"] = g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    feat_cols = [c for c in f.columns if c.startswith(side + "_")]
    match = match.merge(f[["id"] + feat_cols], on="id", how="left")
""")
code("""
# ---- Squad strength: mean FIFA overall rating and potential of the team's line-up in its PREVIOUS match,
#      using each player's latest rating dated before the current match (no match-day information is used).
pa = pd.read_sql("SELECT player_api_id, date, overall_rating, potential FROM Player_Attributes", con)
pa["date"] = pd.to_datetime(pa["date"]); pa = pa.dropna(subset=["overall_rating"]).sort_values("date")

player_cols = {side: [f"{side}_player_{k}" for k in range(1, 12)] for side in ("home", "away")}
long = []
for side, cols in player_cols.items():
    t = match[["id", "date", f"{side}_team_api_id"] + cols].rename(columns={f"{side}_team_api_id": "team"})
    long.append(t.melt(id_vars=["id", "date", "team"], value_vars=cols, value_name="player_api_id").dropna(subset=["player_api_id"]))
lineups = pd.concat(long)[["id", "date", "team", "player_api_id"]]
lineups["player_api_id"] = lineups["player_api_id"].astype("int64")

# previous match of each team and that match's line-up
team_games = pd.concat([match[["id", "date"]].assign(team=match["home_team_api_id"]),
                        match[["id", "date"]].assign(team=match["away_team_api_id"])]).sort_values(["team", "date", "id"])
team_games["prev_id"] = team_games.groupby("team")["id"].shift(1)
prev = team_games.dropna(subset=["prev_id"]).astype({"prev_id": "int64"})
prev_lineups = prev.merge(lineups[["id", "team", "player_api_id"]].rename(columns={"id": "prev_id"}), on=["prev_id", "team"])

# rating of each of those players as at the current match date
prev_lineups = prev_lineups.sort_values("date")
rated = pd.merge_asof(prev_lineups, pa.rename(columns={"date": "rating_date"}), left_on="date", right_on="rating_date",
                      by="player_api_id", direction="backward", allow_exact_matches=False)
squad = rated.groupby(["id", "team"]).agg(squad_overall=("overall_rating", "mean"), squad_potential=("potential", "mean"),
                                          squad_rated=("overall_rating", "count")).reset_index()
for side in ("home", "away"):
    s = squad.rename(columns={"team": f"{side}_team_api_id", "squad_overall": f"{side}_squad_overall",
                              "squad_potential": f"{side}_squad_potential", "squad_rated": f"{side}_squad_rated"})
    match = match.merge(s, on=["id", f"{side}_team_api_id"], how="left")
match["squad_overall_diff"] = match["home_squad_overall"] - match["away_squad_overall"]
match[["home_squad_rated", "away_squad_rated"]].describe().T
""")
code("""
# ---- Head-to-head: previous meetings between the two teams (either venue), from the home team's perspective
pair = match[["id", "date", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"]].copy()
pair["key"] = [tuple(sorted(x)) for x in zip(pair["home_team_api_id"], pair["away_team_api_id"])]
h2h_win, h2h_gd, h2h_n = [], [], []
history = {}
for r in pair.itertuples(index=False):
    past = history.get(r.key, [])
    if past:
        res = [(1.0 if (gf > ga) else 0.0, gf - ga) for (t, gf, ga) in
               [(t, g1, g2) if t == r.home_team_api_id else (t, g2, g1) for (t, g1, g2) in past]]
        h2h_win.append(np.mean([x[0] for x in res])); h2h_gd.append(np.mean([x[1] for x in res])); h2h_n.append(len(res))
    else:
        h2h_win.append(np.nan); h2h_gd.append(np.nan); h2h_n.append(0)
    history.setdefault(r.key, []).append((r.home_team_api_id, r.home_team_goal, r.away_team_goal))
match["h2h_win_ratio"], match["h2h_goal_diff"], match["h2h_n"] = h2h_win, h2h_gd, h2h_n
""")
code("""
FEATURES = (["elo_home", "elo_away", "elo_diff"]
            + [f"{s}_{c}_{w}" for s in ("home", "away") for w in (5, 10) for c in ("gf", "ga", "pts")]
            + ["home_squad_overall", "away_squad_overall", "home_squad_potential", "away_squad_potential", "squad_overall_diff"]
            + ["h2h_win_ratio", "h2h_goal_diff", "h2h_n"])
missing = match[FEATURES].isna().mean().rename("missing share").to_frame()
display(Markdown(f"**{len(FEATURES)} features.** Missing values (for example teams with no previous matches) are imputed with the training-set median inside each model pipeline."))
missing.to_csv(OUT / "table_feature_missing.csv")
missing.style.format("{:.1%}")
""")

code("lap('3. feature engineering')")
md("## 4. Chronological split (70% train, 15% validation, 15% test)")
code("""
n = len(match)
i_tr, i_va = int(n * 0.70), int(n * 0.85)
part = np.array(["train"] * i_tr + ["validation"] * (i_va - i_tr) + ["test"] * (n - i_va))
match["partition"] = part
X, y = match[FEATURES], match["outcome"].to_numpy()
tr, va, te = (part == "train"), (part == "validation"), (part == "test")
split_info = match.groupby("partition", sort=False).agg(matches=("id", "count"), first=("date", "min"), last=("date", "max"))
split_info.to_csv(OUT / "table_split.csv")
split_info
""")

md("## 5. Scoring rules")
code("""
def rps(y_true, p):
    \"\"\"Ranked Probability Score for ordered outcomes H < D < A (Epstein, 1969). Lower is better.\"\"\"
    p = np.asarray(p, dtype=float)
    o = np.eye(3)[np.asarray(y_true)]
    return np.mean(np.sum((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2, axis=1) / 2.0)

rps_scorer = make_scorer(lambda yt, p: -rps(yt, p), response_method="predict_proba")

def scores(y_true, p):
    return {"RPS": rps(y_true, p), "Log-Loss": log_loss(y_true, p, labels=[0, 1, 2]),
            "Accuracy": accuracy_score(y_true, np.argmax(p, 1))}

P_MKT = match[["p_mkt_H", "p_mkt_D", "p_mkt_A"]].to_numpy()
pd.DataFrame({k: scores(y[m], P_MKT[m]) for k, m in [("train", tr), ("validation", va), ("test", te)]}).T.style.format("{:.4f}")
""")

code("lap('4. split and scoring')")
md("## 6. Model tuning (5-fold expanding-window time-series cross-validation inside the training set)")
code("""
tscv = TimeSeriesSplit(n_splits=5)
w_train = compute_sample_weight("balanced", y[tr])

candidates = {
    "Logistic Regression": (Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                            {"clf__C": [0.01, 0.1, 1.0, 10.0]}, False),
    "Random Forest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))]),
                      {"clf__n_estimators": [300], "clf__max_depth": [6, 10, None],
                       "clf__min_samples_split": [20, 50], "clf__max_features": ["sqrt", 0.5]}, False),
    "XGBoost": (Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("clf", XGBClassifier(objective="multi:softprob", eval_metric="mlogloss",
                                                random_state=SEED, n_jobs=-1, tree_method="hist"))]),
                {"clf__learning_rate": [0.03, 0.1], "clf__n_estimators": [200, 400], "clf__max_depth": [3, 5],
                 "clf__subsample": [0.8], "clf__colsample_bytree": [0.8]}, True),
}
tuned, cv_rows = {}, []
for name, (pipe, grid, needs_weight) in candidates.items():
    gs = GridSearchCV(pipe, grid, scoring=rps_scorer, cv=tscv, n_jobs=1)
    fit_kw = {"clf__sample_weight": w_train} if needs_weight else {}
    gs.fit(X[tr], y[tr], **fit_kw)
    tuned[name] = gs.best_estimator_
    cv_rows.append({"model": name, "best params": gs.best_params_, "CV RPS": -gs.best_score_,
                    "configurations": len(gs.cv_results_["params"]), "fits": len(gs.cv_results_["params"]) * tscv.get_n_splits() + 1,
                    "mean seconds per fit": float(np.mean(gs.cv_results_["mean_fit_time"])), "refit seconds": gs.refit_time_})
cv_table = pd.DataFrame(cv_rows); cv_table.to_csv(OUT / "table_cv.csv", index=False)
cv_table
""")

code("lap('5. hyperparameter tuning')")
md("## 7. Calibration and model selection on the validation set")
code("""
from sklearn.base import clone
val_rows, final = [], {}
for name, est in tuned.items():
    needs_weight = candidates[name][2]
    fit_kw = {"clf__sample_weight": w_train} if needs_weight else {}
    variants = {"uncalibrated": est}
    if name != "Logistic Regression":
        for method in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(clone(est), method=method, cv=tscv, ensemble=True)
            cal.fit(X[tr], y[tr], **fit_kw)
            variants[method] = cal
    for vname, model in variants.items():
        p = model.predict_proba(X[va])
        val_rows.append({"model": name, "calibration": vname, **scores(y[va], p)})
    best_v = min(variants, key=lambda v: rps(y[va], variants[v].predict_proba(X[va])))
    final[name] = (best_v, variants[best_v])
val_table = pd.DataFrame(val_rows); val_table.to_csv(OUT / "table_validation.csv", index=False)
SELECTED = min(final, key=lambda m: rps(y[va], final[m][1].predict_proba(X[va])))
display(Markdown(f"**Model selected on the validation set: {SELECTED} ({final[SELECTED][0]})**"))
val_table.style.format({"RPS": "{:.4f}", "Log-Loss": "{:.4f}", "Accuracy": "{:.4f}"})
""")

code("lap('6. calibration and selection')")
md("## 8. Test-set evaluation with bootstrap 95% confidence intervals (RQ2)")
code("""
rng = np.random.default_rng(SEED)
B = 1000
idx_te = np.where(te)[0]
boot = rng.integers(0, len(idx_te), size=(B, len(idx_te)))
test_probs = {name: m.predict_proba(X[te]) for name, (_, m) in final.items()}
test_probs["Market consensus"] = P_MKT[te]
rows = []
for name, p in test_probs.items():
    point = scores(y[te], p)
    draws = pd.DataFrame([scores(y[te][b], p[b]) for b in boot])
    ci = draws.quantile([0.025, 0.975])
    rows.append({"model": name, **{k: f"{v:.4f} [{ci.loc[0.025, k]:.4f}, {ci.loc[0.975, k]:.4f}]" for k, v in point.items()}})
test_table = pd.DataFrame(rows); test_table.to_csv(OUT / "table_test.csv", index=False)
test_table
""")
md("Paired bootstrap of the difference in RPS between each model and the market consensus (positive = the market forecast better). The same resampled matches are used for both forecasts, so the comparison accounts for match difficulty.")
code("""
diff_rows = []
for name in final:
    d = np.array([rps(y[te][b], test_probs[name][b]) - rps(y[te][b], P_MKT[te][b]) for b in boot])
    point = rps(y[te], test_probs[name]) - rps(y[te], P_MKT[te])
    diff_rows.append({"model": name, "RPS difference (model - market)": point,
                      "95% CI low": np.percentile(d, 2.5), "95% CI high": np.percentile(d, 97.5),
                      "share of resamples where model beats market": np.mean(d < 0)})
diff_table = pd.DataFrame(diff_rows); diff_table.to_csv(OUT / "table_rps_diff_vs_market.csv", index=False)
diff_table.style.format({c: "{:.4f}" for c in diff_table.columns if c != "model"})
""")
code("""
league_rows = []
for lg in sorted(match["league"].unique()):
    m = (match.loc[te, "league"] == lg).to_numpy()
    league_rows.append({"league": lg, "matches": int(m.sum()),
                        **{name: rps(y[te][m], p[m]) for name, p in test_probs.items()}})
league_table = pd.DataFrame(league_rows).set_index("league"); league_table.to_csv(OUT / "table_rps_by_league.csv")
league_table.style.format("{:.4f}", subset=list(test_probs))
""")
code("""
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for k, (ax, lab) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    for name, p in test_probs.items():
        fx, fy = calibration_curve((y[te] == k).astype(int), p[:, k], n_bins=10, strategy="quantile")
        ax.plot(fy, fx, marker="o", ms=3, label=name)
    ax.set_title(lab); ax.set_xlabel("Predicted probability")
axes[0].set_ylabel("Observed frequency"); axes[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_calibration.png", dpi=200)
""")

code("lap('7. test evaluation and bootstrap')")
md("## 9. Permutation importance (best tree-based model)")
code("""
tree_name = min(["Random Forest", "XGBoost"], key=lambda m: rps(y[va], final[m][1].predict_proba(X[va])))
pi = permutation_importance(final[tree_name][1], X[te], y[te], scoring=rps_scorer, n_repeats=10, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({"feature": FEATURES, "importance (RPS increase)": pi.importances_mean, "std": pi.importances_std}) \\
        .sort_values("importance (RPS increase)", ascending=False)
imp.to_csv(OUT / "table_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 5))
top = imp.head(12)[::-1]
ax.barh(top["feature"], top["importance (RPS increase)"], xerr=top["std"], color="#2E75B6")
ax.set_xlabel("Mean increase in test RPS when permuted"); ax.set_title(f"Permutation importance: {tree_name}")
fig.tight_layout(); fig.savefig(OUT / "fig_importance.png", dpi=200)
imp.head(12)
""")

code("lap('8. permutation importance')")
md("""
## 10. Market efficiency test (RQ1, RQ3)
Edge = model probability minus consensus probability. A one-unit value bet is placed whenever the edge exceeds the threshold.
Primary settlement: average odds across bookmakers. Sensitivity: best available odds.
""")
code("""
P_SEL = test_probs[SELECTED]
T = match.loc[te].reset_index(drop=True)
edge = P_SEL - P_MKT[te]
odds_avg = T[["odds_avg_H", "odds_avg_D", "odds_avg_A"]].to_numpy()
odds_max = T[["odds_max_H", "odds_max_D", "odds_max_A"]].to_numpy()
won = np.eye(3)[y[te]].astype(bool)

edge_summary = pd.DataFrame({lab: pd.Series(edge[:, k]).describe(percentiles=[.05, .5, .95]) for k, lab in enumerate(["Home", "Draw", "Away"])}).T
display(edge_summary.style.format("{:.4f}"))
fig, ax = plt.subplots(figsize=(7, 3.5))
for k, lab in enumerate(["Home", "Draw", "Away"]):
    ax.hist(edge[:, k] * 100, bins=60, histtype="step", label=lab)
for t in (3, 5, 7): ax.axvline(t, c="grey", ls=":", lw=1)
ax.set_xlabel("Edge (percentage points)"); ax.set_ylabel("Outcomes"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig_edges.png", dpi=200)
""")
code("""
N_PERM = 10_000
def roi(mask, odds):
    stake = mask.sum()
    return np.nan if stake == 0 else (np.where(mask & won, odds - 1, 0).sum() - (mask & ~won).sum()) / stake

def perm_pvalue(mask, odds, observed, stratified):
    rng = np.random.default_rng(SEED)
    flat_won, flat_odds = won.ravel(), odds.ravel()
    ok = np.isfinite(flat_odds)
    hits = np.empty(N_PERM)
    if stratified:   # same number of bets per outcome type (home/draw/away) as the strategy
        per_k = mask.sum(axis=0)
        pools = [np.where(ok & (np.arange(flat_odds.size) % 3 == k))[0] for k in range(3)]
    for i in range(N_PERM):
        if stratified:
            pick = np.concatenate([rng.choice(pools[k], per_k[k], replace=False) for k in range(3) if per_k[k]])
        else:
            pick = rng.choice(np.where(ok)[0], mask.sum(), replace=False)
        hits[i] = (np.where(flat_won[pick], flat_odds[pick] - 1, -1.0)).mean()
    return (np.sum(hits >= observed) + 1) / (N_PERM + 1), hits.mean(), np.percentile(hits, 97.5)

rows = []
for settle, odds in [("average odds", odds_avg), ("best odds", odds_max)]:
    for t in (0.03, 0.05, 0.07):
        mask = edge > t
        r = roi(mask, odds)
        p_r, _, _ = perm_pvalue(mask, odds, r, False)
        p_s, null_mean, null_q975 = perm_pvalue(mask, odds, r, True)
        rows.append({"settlement": settle, "threshold (pp)": int(t * 100), "bets": int(mask.sum()),
                     "home/draw/away bets": "/".join(str(int(v)) for v in mask.sum(axis=0)),
                     "hit rate": (mask & won).sum() / max(mask.sum(), 1), "profit (units)": r * mask.sum(),
                     "ROI": r, "null mean ROI (stratified)": null_mean, "null 97.5th pct": null_q975,
                     "p (random)": p_r, "p (stratified)": p_s})
bet_table = pd.DataFrame(rows); bet_table.to_csv(OUT / "table_betting.csv", index=False)
bet_table.style.format({"hit rate": "{:.1%}", "profit (units)": "{:.2f}", "ROI": "{:.2%}", "null mean ROI (stratified)": "{:.2%}",
                        "null 97.5th pct": "{:.2%}", "p (random)": "{:.4f}", "p (stratified)": "{:.4f}"})
""")
code("""
rows = []
for lg in sorted(T["league"].unique()):
    lm = (T["league"] == lg).to_numpy()[:, None]
    for t in (0.03, 0.05, 0.07):
        mask = (edge > t) & lm
        rows.append({"league": lg, "threshold (pp)": int(t * 100), "bets": int(mask.sum()), "ROI (average odds)": roi(mask, odds_avg)})
league_bets = pd.DataFrame(rows).pivot(index="league", columns="threshold (pp)", values=["bets", "ROI (average odds)"])
league_bets.to_csv(OUT / "table_betting_by_league.csv")
league_bets
""")
code("""
fig, ax = plt.subplots(figsize=(8, 3.8))
for t in (0.03, 0.05, 0.07):
    mask = edge > t
    pnl = np.where(mask & won, odds_avg - 1, 0).sum(1) - (mask & ~won).sum(1)
    ax.plot(T["date"], np.cumsum(pnl), label=f"threshold {int(t*100)} pp")
ax.axhline(0, c="grey", lw=1); ax.set_ylabel("Cumulative profit (units)"); ax.legend()
ax.set_title(f"Cumulative profit on the test set ({SELECTED}, average odds)")
fig.tight_layout(); fig.savefig(OUT / "fig_cumulative_profit.png", dpi=200)
""")

md("Outcome mix in each partition, and returns split by the outcome type that was backed (average odds).")
code("""
mix = match.groupby("partition", sort=False)["outcome"].value_counts(normalize=True).unstack().rename(columns={0: "home", 1: "draw", 2: "away"})
mix.to_csv(OUT / "table_outcome_mix_by_partition.csv")
display(mix.style.format("{:.1%}"))
rows = []
for t in (0.03, 0.05, 0.07):
    for k, lab in enumerate(["home", "draw", "away"]):
        mask = np.zeros_like(won); mask[:, k] = edge[:, k] > t
        rows.append({"threshold (pp)": int(t * 100), "outcome backed": lab, "bets": int(mask.sum()),
                     "hit rate": (mask & won).sum() / max(mask.sum(), 1), "ROI (average odds)": roi(mask, odds_avg)})
by_type = pd.DataFrame(rows); by_type.to_csv(OUT / "table_betting_by_outcome.csv", index=False)
by_type.style.format({"hit rate": "{:.1%}", "ROI (average odds)": "{:.2%}"})
""")

code("lap('9. betting simulation and permutation tests')")
md("## 11. Reproducibility record")
code("""
record = {"seed": SEED, "matches": int(len(match)), "features": FEATURES, "selected_model": SELECTED,
          "selected_calibration": final[SELECTED][0], "min_books": MIN_BOOKS, "elo": [ELO_START, ELO_K],
          "bookmakers": BOOKS, "permutations": N_PERM, "bootstrap": B}
(OUT / "run_record.json").write_text(json.dumps(record, indent=2))
pd.Series(record).to_frame("value")
""")
md("### Run time by stage")
code("""
lap("10. reproducibility record")
timing = pd.Series(TIMER, name="seconds").to_frame()
timing.loc["Total"] = timing["seconds"].sum()
timing.to_csv(OUT / "table_runtime.csv")
display(Markdown(f"Processor cores available: **{os.cpu_count()}**; machine: **{platform.processor() or platform.machine()}**"))
timing.style.format("{:.1f}")
""")

md("""
# 12. Post-review checks (added after supervisor feedback, November 2026)
Everything above this point is the original analysis and is unchanged. The checks below were added after the test results had been seen,
so they are **sensitivity analyses and revisions, not confirmatory tests**. Original hyperparameters and thresholds are kept unless a cell says otherwise.
""")
code("""
from bisect import bisect_left
from sklearn.base import clone
_review_start = time.perf_counter()
R = OUT / "review"; R.mkdir(exist_ok=True)
# outcome labels and per-match scores used throughout this section
LAB = ["Home", "Draw", "Away"]
def rps_each(y_true, p):
    p = np.asarray(p, dtype=float); o = np.eye(3)[np.asarray(y_true)]
    return np.sum((np.cumsum(p, 1)[:, :2] - np.cumsum(o, 1)[:, :2]) ** 2, axis=1) / 2.0
def ll_each(y_true, p):
    return -np.log(np.clip(np.asarray(p)[np.arange(len(y_true)), y_true], 1e-15, 1))
# calendar-week blocks for the dependence-aware bootstrap (Monday to Sunday)
week_id = T["date"].dt.to_period("W").astype(str).to_numpy()
weeks = pd.unique(week_id); week_rows = [np.where(week_id == w)[0] for w in weeks]
rng_b = np.random.default_rng(SEED + 1)
BLOCK = [np.concatenate([week_rows[j] for j in rng_b.integers(0, len(weeks), len(weeks))]) for _ in range(B)]
pd.Series({"test matches": len(T), "calendar-week blocks": len(weeks), "matches per week, median": int(np.median([len(r) for r in week_rows])),
           "matches per week, max": int(max(len(r) for r in week_rows))}).to_frame("value")
""")

md("## 12.1 Data-quality audit")
code("""
raw = pd.read_sql("SELECT * FROM Match", con)
raw = raw[raw["league_id"].isin(target_ids["league_id"])].copy()
raw["date"] = pd.to_datetime(raw["date"])
o_cols = [c for cols in odds_cols.values() for c in cols]
vals = raw[o_cols].apply(pd.to_numeric, errors="coerce")
partial = sum(int((raw[c].notna().sum(axis=1).between(1, 2)).sum()) for c in odds_cols.values())
n_books_raw = sum(((raw[c].notna().all(axis=1)) & (raw[c].apply(pd.to_numeric, errors="coerce") > 1).all(axis=1)).astype(int) for c in odds_cols.values())
pa_raw = pd.read_sql("SELECT player_api_id, date, overall_rating, potential FROM Player_Attributes", con)
pa_dup = pa_raw.duplicated(["player_api_id", "date"], keep=False)
audit = {
 "five-league matches before cleaning": len(raw),
 "duplicate id": int(raw["id"].duplicated().sum()),
 "duplicate match_api_id": int(raw["match_api_id"].duplicated().sum()),
 "duplicate date + home team + away team": int(raw.duplicated(["date", "home_team_api_id", "away_team_api_id"]).sum()),
 "duplicate season + home team + away team": int(raw.duplicated(["season", "home_team_api_id", "away_team_api_id"]).sum()),
 "home team equals away team": int((raw["home_team_api_id"] == raw["away_team_api_id"]).sum()),
 "missing date, team or goals": int(raw[["date", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"]].isna().any(axis=1).sum()),
 "negative goals": int(((raw["home_team_goal"] < 0) | (raw["away_team_goal"] < 0)).sum()),
 "non-integer goals": int(((raw["home_team_goal"] % 1 != 0) | (raw["away_team_goal"] % 1 != 0)).sum()),
 "maximum goals by one team": int(raw[["home_team_goal", "away_team_goal"]].max().max()),
 "individual odds values at or below 1.00": int((vals <= 1).sum().sum()),
 "non-numeric odds values": int((raw[o_cols].notna() & vals.isna()).sum().sum()),
 "bookmaker triplets with only one or two of three prices": partial,
 "matches with 0 complete bookmakers": int((n_books_raw == 0).sum()),
 "matches with 1 complete bookmaker": int((n_books_raw == 1).sum()),
 "matches with 2 complete bookmakers": int((n_books_raw == 2).sum()),
 "matches with 3 or 4 complete bookmakers": int(n_books_raw.between(3, 4).sum()),
 "matches with 5 or more complete bookmakers": int((n_books_raw >= 5).sum()),
 "minimum complete bookmakers among retained matches": int(match["n_books"].min()),
 "retained matches with mean overround at or below 0": int((match["overround_mean"] <= 0).sum()),
 "lowest and highest mean overround retained (%)": f"{match['overround_mean'].min()*100:.2f} to {match['overround_mean'].max()*100:.2f}",
 "line-up rows (team, match, player)": len(lineups),
 "duplicate player within a team line-up": int(lineups.duplicated(["id", "team", "player_api_id"]).sum()),
 "previous-line-up rows before rating join": len(prev_lineups),
 "previous-line-up rows after rating join": len(rated),
 "rows added by rating join": len(rated) - len(prev_lineups),
 "team-match rows in squad table": len(squad),
 "duplicate team-match rows in squad table": int(squad.duplicated(["id", "team"]).sum()),
 "matches after all feature joins": len(match),
 "duplicate id after all feature joins": int(match["id"].duplicated().sum()),
 "Player_Attributes rows": len(pa_raw),
 "Player_Attributes rows with missing overall rating (dropped)": int(pa_raw["overall_rating"].isna().sum()),
 "kept rating rows with missing potential": int(pa["potential"].isna().sum()),
 "rating rows sharing a player and date with another row": int(pa_dup.sum()),
 "players in previous line-ups without an eligible rating": int(rated["overall_rating"].isna().sum()),
 "ratings outside 0 to 100": int(((pa["overall_rating"] < 0) | (pa["overall_rating"] > 100)).sum()),
}
for side in ("home", "away"):
    audit[f"{side} team-matches with no previous line-up or no rated player"] = int(match[f"{side}_squad_rated"].fillna(0).eq(0).sum())
    audit[f"{side} team-matches with fewer than 11 rated players (excluding zero)"] = int(match[f"{side}_squad_rated"].between(1, 10).sum())
audit_table = pd.Series(audit, name="value").to_frame(); audit_table.to_csv(R / "data_audit.csv")
audit_table
""")

md("## 12.2 Team entry, promotion and re-entry (Elo and form histories)")
code("""
g = pd.concat([match[["date", "season", "home_team_api_id"]].rename(columns={"home_team_api_id": "team"}),
               match[["date", "season", "away_team_api_id"]].rename(columns={"away_team_api_id": "team"})]).sort_values("date")
seasons = sorted(match["season"].unique()); s_ix = {s: i for i, s in enumerate(seasons)}
t_seasons = g.groupby("team")["season"].apply(lambda s: sorted({s_ix[x] for x in s}))
entry_rows = {"teams": len(t_seasons),
              "teams present from the first season": int(t_seasons.apply(lambda v: v[0] == 0).sum()),
              "teams first appearing in a later season (start at 1,500)": int(t_seasons.apply(lambda v: v[0] > 0).sum()),
              "teams with a gap of at least one season (rating carried over)": int(t_seasons.apply(lambda v: len(v) != v[-1] - v[0] + 1).sum()),
              "matches in the first season": int((match["season"] == seasons[0]).sum())}
pd.Series(entry_rows, name="value").to_frame().to_csv(R / "team_entry.csv")
pd.Series(entry_rows, name="value").to_frame()
""")

md("## 12.3 Fold boundaries, same-day handling and feature availability in folds")
code("""
Xtr = X[tr]; date_tr = match.loc[tr, "date"].reset_index(drop=True); y_tr = y[tr]
fold_rows = []
for k, (a, b) in enumerate(tscv.split(Xtr), 1):
    fold_rows.append({"fold": k, "fit matches": len(a), "fit first": date_tr.iloc[a[0]].date(), "fit last": date_tr.iloc[a[-1]].date(),
                      "scored matches": len(b), "scored first": date_tr.iloc[b[0]].date(), "scored last": date_tr.iloc[b[-1]].date(),
                      "fit matches sharing the first scored date": int((date_tr.iloc[a] == date_tr.iloc[b[0]]).sum()),
                      "scored matches on that shared date": int((date_tr.iloc[b] == date_tr.iloc[b[0]]).sum()) if (date_tr.iloc[a] == date_tr.iloc[b[0]]).any() else 0,
                      "features entirely missing in fit block": int(Xtr.iloc[a].isna().all().sum()),
                      "home/draw/away share in fit block": " / ".join(f"{(y_tr[a] == c).mean():.1%}" for c in range(3))})
fold_table = pd.DataFrame(fold_rows); fold_table.to_csv(R / "fold_boundaries.csv", index=False)
bound = []
for nm, left, right in [("train / validation", tr, va), ("validation / test", va, te)]:
    d0 = match.loc[right, "date"].min()
    bound.append({"boundary": nm, "date": d0.date(), "matches on date, earlier partition": int((match.loc[left, "date"] == d0).sum()),
                  "matches on date, later partition": int((match.loc[right, "date"] == d0).sum())})
bound_table = pd.DataFrame(bound); bound_table.to_csv(R / "split_boundaries.csv", index=False)
display(bound_table)
cal_check = {name: {"calibrated folds": len(m.calibrated_classifiers_), "class": type(m).__name__, "method": m.method, "ensemble": m.ensemble}
             for name, (v, m) in final.items() if v != "uncalibrated"}
display(pd.DataFrame(cal_check).T)
fold_table
""")

md("## 12.4 Class-weight routing: XGBoost tuned with weights recomputed inside each fold")
code("""
xgb_grid = candidates["XGBoost"][1]
from sklearn.model_selection import ParameterGrid
rows = []
for params in ParameterGrid(xgb_grid):
    sc_fold, sc_glob = [], []
    for a, b in tscv.split(Xtr):
        for store, w in ((sc_fold, compute_sample_weight("balanced", y_tr[a])), (sc_glob, w_train[a])):
            est = clone(candidates["XGBoost"][0]).set_params(**params)
            est.fit(Xtr.iloc[a], y_tr[a], clf__sample_weight=w)
            store.append(rps(y_tr[b], est.predict_proba(Xtr.iloc[b])))
    rows.append({**{k.replace("clf__", ""): v for k, v in params.items()}, "CV RPS, weights from whole training set": np.mean(sc_glob), "CV RPS, weights recomputed per fold": np.mean(sc_fold)})
wt = pd.DataFrame(rows); wt.to_csv(R / "xgb_weight_routing.csv", index=False)
wt.sort_values("CV RPS, weights recomputed per fold").head(4)
""")

md("""
## 12.5 Weighted versus unweighted pipelines, with the same calibration opportunity for all three algorithms
Each algorithm is retuned on the training set under each weighting rule and offered uncalibrated, Platt and isotonic versions.
Choices are made on the validation set. Test results are a **revision made after the original test had been inspected**.
""")
code("""
def make_pipe(name, weighted):
    p = clone(candidates[name][0])
    if name != "XGBoost": p.set_params(clf__class_weight="balanced" if weighted else None)
    return p
rev_rows, rev_models = [], {}
for name in candidates:
    for weighted in (True, False):
        fit_kw = {"clf__sample_weight": w_train} if (weighted and name == "XGBoost") else {}
        gs = GridSearchCV(make_pipe(name, weighted), candidates[name][1], scoring=rps_scorer, cv=tscv, n_jobs=1)
        gs.fit(X[tr], y[tr], **fit_kw)
        variants = {"uncalibrated": gs.best_estimator_}
        for method in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(clone(gs.best_estimator_), method=method, cv=tscv, ensemble=True)
            cal.fit(X[tr], y[tr], **fit_kw); variants[method] = cal
        for v, m in variants.items():
            pv, pt = m.predict_proba(X[va]), m.predict_proba(X[te])
            rev_models[(name, weighted, v)] = (m, pv, pt)
            rev_rows.append({"model": name, "weights": "balanced" if weighted else "none", "calibration": v,
                             "best params": str({k.replace("clf__", ""): w for k, w in gs.best_params_.items()}), "CV RPS": -gs.best_score_,
                             "validation RPS": rps(y[va], pv), "validation Log-Loss": log_loss(y[va], pv, labels=[0, 1, 2]),
                             "test RPS": rps(y[te], pt), "test Log-Loss": log_loss(y[te], pt, labels=[0, 1, 2]), "test accuracy": accuracy_score(y[te], pt.argmax(1))})
rev = pd.DataFrame(rev_rows); rev.to_csv(R / "weighted_vs_unweighted.csv", index=False)
best_lab = rev.loc[rev["validation RPS"].idxmin()]
best_by_algo = rev.loc[rev.groupby("model")["validation RPS"].idxmin()]
best_by_algo.to_csv(R / "revised_selected_by_algorithm.csv", index=False)
display(Markdown(f"Lowest validation RPS across all pipelines: **{best_lab['model']}, weights {best_lab['weights']}, {best_lab['calibration']}**"))
display(best_by_algo[["model", "weights", "calibration", "validation RPS", "test RPS", "test Log-Loss", "test accuracy"]])
rev.drop(columns="best params")
""")

md("## 12.6 Uncertainty: match-level and calendar-week block bootstrap, model-to-market and model-to-model")
code("""
yt = y[te]
fc = {"Logistic Regression": test_probs["Logistic Regression"], "Random Forest": test_probs["Random Forest"], "XGBoost": test_probs["XGBoost"], "Market consensus": P_MKT[te]}
rps_m = {k: rps_each(yt, p) for k, p in fc.items()}; ll_m = {k: ll_each(yt, p) for k, p in fc.items()}
def ci(fn, sets):
    v = np.array([fn(ix) for ix in sets]); return np.percentile(v, 2.5), np.percentile(v, 97.5), v
rows = []
for k in fc:
    for nm, sets in (("match", boot), ("week block", BLOCK)):
        lo, hi, _ = ci(lambda ix: rps_m[k][ix].mean(), sets); lo2, hi2, _ = ci(lambda ix: ll_m[k][ix].mean(), sets)
        rows.append({"forecast": k, "resampling": nm, "RPS": rps_m[k].mean(), "RPS low": lo, "RPS high": hi, "Log-Loss": ll_m[k].mean(), "LL low": lo2, "LL high": hi2})
pd.DataFrame(rows).to_csv(R / "bootstrap_levels.csv", index=False); boot_levels = pd.DataFrame(rows)
pairs = [("Logistic Regression", "Market consensus"), ("Random Forest", "Market consensus"), ("XGBoost", "Market consensus"),
         ("XGBoost", "Random Forest"), ("XGBoost", "Logistic Regression"), ("Random Forest", "Logistic Regression")]
rows = []
for a, b in pairs:
    d = rps_m[a] - rps_m[b]; dl = ll_m[a] - ll_m[b]
    for nm, sets in (("match", boot), ("week block", BLOCK)):
        lo, hi, v = ci(lambda ix: d[ix].mean(), sets); lo2, hi2, _ = ci(lambda ix: dl[ix].mean(), sets)
        rows.append({"A": a, "B": b, "resampling": nm, "RPS A minus B": d.mean(), "low": lo, "high": hi, "resamples A better": int((v < 0).sum()), "resamples": len(sets),
                     "Log-Loss A minus B": dl.mean(), "LL low": lo2, "LL high": hi2})
boot_pairs = pd.DataFrame(rows); boot_pairs.to_csv(R / "bootstrap_pairs.csv", index=False)
boot_pairs
""")

md("## 12.7 Reliability bins, classwise calibration error and calibration in the large (test set)")
code("""
def quantile_bins(y01, p, n_bins=10):
    edges = np.percentile(p, np.linspace(0, 100, n_bins + 1)); edges[-1] += 1e-8
    ids = np.searchsorted(edges[1:-1], p)
    out = []
    for b in np.unique(ids):
        m = ids == b; n = int(m.sum()); k = y01[m].mean()
        z = 1.96; c = (k + z*z/(2*n)) / (1 + z*z/n); h = z * np.sqrt(k*(1-k)/n + z*z/(4*n*n)) / (1 + z*z/n)
        out.append({"bin": int(b), "n": n, "pred low": p[m].min(), "pred high": p[m].max(), "mean predicted": p[m].mean(), "observed": k, "wilson low": c-h, "wilson high": c+h})
    return pd.DataFrame(out)
def ece(y01, p):
    t = quantile_bins(y01, p); return np.sum(t["n"] / t["n"].sum() * np.abs(t["mean predicted"] - t["observed"]))
bin_rows, cal_rows = [], []
for name, p in fc.items():
    for k, lab in enumerate(LAB):
        y01 = (yt == k).astype(float); t = quantile_bins(y01, p[:, k]); t.insert(0, "outcome", lab); t.insert(0, "forecast", name); bin_rows.append(t)
        low, high, _ = ci(lambda ix: ece(y01[ix], p[ix, k]), BLOCK)
        l2, h2, _ = ci(lambda ix: p[ix, k].mean() - y01[ix].mean(), BLOCK)
        cal_rows.append({"forecast": name, "outcome": lab, "bins used": len(t), "smallest bin": int(t["n"].min()), "largest bin": int(t["n"].max()),
                         "pred 5th pct": np.percentile(p[:, k], 5), "pred 95th pct": np.percentile(p[:, k], 95), "ECE": ece(y01, p[:, k]), "ECE low": low, "ECE high": high,
                         "mean predicted minus observed": p[:, k].mean() - y01.mean(), "gap low": l2, "gap high": h2,
                         "bins whose Wilson interval excludes the mean prediction": int(((t["mean predicted"] < t["wilson low"]) | (t["mean predicted"] > t["wilson high"])).sum())})
bins_all = pd.concat(bin_rows); bins_all.to_csv(R / "reliability_bins.csv", index=False)
cal_tab = pd.DataFrame(cal_rows); cal_tab.to_csv(R / "calibration_summary.csv", index=False)
# paired differences in mean classwise ECE between forecasts
mean_ece = lambda name, ix: np.mean([ece((yt[ix] == k).astype(float), fc[name][ix, k]) for k in range(3)])
rows = []
for a, b in [("XGBoost", "Random Forest"), ("XGBoost", "Logistic Regression"), ("XGBoost", "Market consensus"), ("Random Forest", "Logistic Regression")]:
    v = np.array([mean_ece(a, ix) - mean_ece(b, ix) for ix in BLOCK[:500]])
    rows.append({"A": a, "B": b, "mean classwise ECE A": mean_ece(a, np.arange(len(yt))), "mean classwise ECE B": mean_ece(b, np.arange(len(yt))), "difference low": np.percentile(v, 2.5), "difference high": np.percentile(v, 97.5)})
ece_pairs = pd.DataFrame(rows); ece_pairs.to_csv(R / "calibration_pairs.csv", index=False)
fig, axes = plt.subplots(3, 1, figsize=(7.2, 11.5))
sty = {"Logistic Regression": ("#C55A11", "s"), "Random Forest": ("#548235", "^"), "XGBoost": ("#1F4E79", "o"), "Market consensus": ("#7F7F7F", "D")}
for k, (ax, lab) in enumerate(zip(axes, ["Home win", "Draw", "Away win"])):
    t = bins_all[bins_all["outcome"] == LAB[k]]
    xs = t[["mean predicted"]].to_numpy(); lim = (max(0, xs.min() - .05), min(1, xs.max() + .05))
    ax.plot(lim, lim, ls="--", c="grey", lw=1, label="Perfect calibration")
    for j, name in enumerate(fc):
        q = t[t["forecast"] == name]; off = (j - 1.5) * 0.002
        ax.errorbar(q["mean predicted"] + off, q["observed"], yerr=[q["observed"] - q["wilson low"], q["wilson high"] - q["observed"]], color=sty[name][0], marker=sty[name][1], ms=4.5, lw=1.1, elinewidth=.6, capsize=1.5, label=name)
    ax.set_title(lab, fontsize=12, fontweight="bold"); ax.set_xlabel("Mean predicted probability in bin", fontsize=10.5); ax.set_ylabel("Observed frequency", fontsize=10.5); ax.tick_params(labelsize=10); ax.grid(alpha=.25)
axes[0].legend(fontsize=9.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fig_calibration_large.png", dpi=200)
display(ece_pairs)
cal_tab
""")

md("## 12.8 Over-weighting of home wins: model, market and actual by league and period")
code("""
def gap_table(df_probs, frame, ys, label):
    rows = []
    groups = [("All leagues", np.ones(len(frame), bool))] + [(lg, (frame["league"] == lg).to_numpy()) for lg in sorted(frame["league"].unique())]
    w = frame["date"].dt.to_period("W").astype(str).to_numpy(); ws = pd.unique(w); wr = [np.where(w == v)[0] for v in ws]
    rg = np.random.default_rng(SEED + 2)
    sets = [np.concatenate([wr[j] for j in rg.integers(0, len(ws), len(ws))]) for _ in range(B)]
    for nm, m in groups:
        mh, kh, ah = df_probs[:, 0], frame["p_mkt_H"].to_numpy(), (ys == 0).astype(float)
        ix0 = np.where(m)[0]
        v = np.array([ (mh[s][m[s]] - kh[s][m[s]]).mean() for s in sets]); v2 = np.array([(mh[s][m[s]] - ah[s][m[s]]).mean() for s in sets]); v3 = np.array([(kh[s][m[s]] - ah[s][m[s]]).mean() for s in sets])
        rows.append({"period": label, "group": nm, "matches": len(ix0), "model mean P(home)": mh[m].mean(), "market mean P(home)": kh[m].mean(), "actual home share": ah[m].mean(),
                     "model minus market": (mh - kh)[m].mean(), "low": np.percentile(v, 2.5), "high": np.percentile(v, 97.5),
                     "model minus actual": (mh - ah)[m].mean(), "model-actual low": np.percentile(v2, 2.5), "model-actual high": np.percentile(v2, 97.5),
                     "market minus actual": (kh - ah)[m].mean(), "market-actual low": np.percentile(v3, 2.5), "market-actual high": np.percentile(v3, 97.5)})
    return rows
V = match.loc[va].reset_index(drop=True)
pv_sel = final[SELECTED][1].predict_proba(X[va])
is_1516 = (T["season"] == "2015/2016").to_numpy()
rows = gap_table(pv_sel, V, y[va], "Validation") + gap_table(P_SEL, T, yt, "Test (all)") \\
     + gap_table(P_SEL[~is_1516], T[~is_1516].reset_index(drop=True), yt[~is_1516], "Test: end of 2014/15") \\
     + gap_table(P_SEL[is_1516], T[is_1516].reset_index(drop=True), yt[is_1516], "Test: 2015/16")
home_gap = pd.DataFrame(rows); home_gap.to_csv(R / "home_gap.csv", index=False)
home_gap
""")

md("""
## 12.9 Does retraining on more recent matches reduce the gap?
Same hyperparameters and isotonic calibration as the selected model, kept chronologically fair: (a) refitted once on training + validation before the test period; (b) refitted at the start of each calendar month using only matches dated before that month.
The weighted and unweighted versions of (b) are compared.
""")
code("""
xgb_params = tuned["XGBoost"].get_params()
def fit_sel(Xfit, yfit, weighted=True):
    base = clone(tuned["XGBoost"])
    cal = CalibratedClassifierCV(base, method="isotonic", cv=TimeSeriesSplit(n_splits=5), ensemble=True)
    kw = {"clf__sample_weight": compute_sample_weight("balanced", yfit)} if weighted else {}
    cal.fit(Xfit, yfit, **kw); return cal
tv = tr | va
p_tv = fit_sel(X[tv], y[tv]).predict_proba(X[te])
def walk(weighted):
    out = np.zeros_like(P_SEL); months = T["date"].dt.to_period("M").astype(str).to_numpy()
    for mth in pd.unique(months):
        start = pd.Period(mth).start_time; fit = (match["date"] < start).to_numpy()
        ix = months == mth
        out[ix] = fit_sel(X[fit], y[fit], weighted).predict_proba(X[te][ix])
    return out, len(pd.unique(months))
p_w, n_months = walk(True); p_wu, _ = walk(False)
p_un = rev_models[("XGBoost", False, "isotonic")][2]
retrain = {"Original selected model": P_SEL, "Unweighted (training set only)": p_un, "Refit once on training + validation": p_tv,
           f"Refit monthly, weighted ({n_months} refits)": p_w, f"Refit monthly, unweighted ({n_months} refits)": p_wu}
RET_PROBS = retrain
rows = []
for nm, p in retrain.items():
    d = rps_each(yt, p) - rps_m["Market consensus"]; lo, hi, _ = ci(lambda ix: d[ix].mean(), BLOCK)
    g = p[:, 0] - P_MKT[te][:, 0]; l2, h2, _ = ci(lambda ix: g[ix].mean(), BLOCK)
    rows.append({"version": nm, "test RPS": rps(yt, p), "RPS minus market": d.mean(), "low": lo, "high": hi, "model mean P(home)": p[:, 0].mean(), "model minus market P(home)": g.mean(), "gap low": l2, "gap high": h2})
retrain_table = pd.DataFrame(rows); retrain_table.to_csv(R / "retraining.csv", index=False)
retrain_table
""")

md("## 12.10 Structure of the bets: same-match bets, profitability condition, and strategy versus null comparability")
code("""
def bet_stats(mask, odds, p):
    n = mask.sum(); pos = p * odds > 1
    both = mask.sum(1)
    return {"bets": int(n), "matches with a bet": int((both > 0).sum()), "matches with two bets": int((both == 2).sum()), "matches with three bets": int((both == 3).sum()),
            "home+draw pairs": int((mask[:, 0] & mask[:, 1]).sum()), "home+away pairs": int((mask[:, 0] & mask[:, 2]).sum()), "draw+away pairs": int((mask[:, 1] & mask[:, 2]).sum()),
            "bets with model P x odds > 1": int((mask & pos).sum()), "share": (mask & pos).sum() / n, "mean model P x odds": (p * odds)[mask].mean()}
rows = []
for settle, odds in (("average", odds_avg), ("best", odds_max)):
    for t in (.03, .05, .07):
        rows.append({"settlement": settle, "threshold": int(t*100), **bet_stats(edge > t, odds, P_SEL)})
struct = pd.DataFrame(rows); struct.to_csv(R / "bet_structure.csv", index=False); display(struct)
# null versus strategy: what is being compared
flat_league = np.repeat(T["league"].to_numpy(), 3); flat_half = np.repeat((T["date"] >= T["date"].iloc[len(T)//2]).to_numpy(), 3); flat_k = np.tile(np.arange(3), len(T))
rows = []
for t in (.03, .05, .07):
    mask = edge > t; per = mask.sum(0)
    for k, lab in enumerate(LAB):
        s, pool = odds_avg[:, k][mask[:, k]], odds_avg[:, k]
        rows.append({"threshold": int(t*100), "outcome": lab, "strategy bets": int(per[k]), "strategy mean odds": s.mean(), "strategy median odds": np.median(s),
                     "null pool mean odds": pool.mean(), "null pool median odds": np.median(pool),
                     "strategy mean market P": P_MKT[te][:, k][mask[:, k]].mean(), "null pool mean market P": P_MKT[te][:, k].mean()})
null_odds = pd.DataFrame(rows); null_odds.to_csv(R / "null_vs_strategy_odds.csv", index=False)
rows = []
for t in (.03, .05, .07):
    mask = edge > t; flat = mask.ravel()
    for lg in sorted(T["league"].unique()):
        rows.append({"threshold": int(t*100), "group": lg, "strategy share": (flat_league[flat] == lg).mean(), "null pool share": (flat_league == lg).mean()})
    rows.append({"threshold": int(t*100), "group": "Second half of test period", "strategy share": flat_half[flat].mean(), "null pool share": flat_half.mean()})
null_mix = pd.DataFrame(rows); null_mix.to_csv(R / "null_vs_strategy_mix.csv", index=False)
display(null_mix)
null_odds
""")

md("""
## 12.11 Null distributions: quantiles of the original random-betting null, and an added fixed-bets simulation
The original null draws different bets (without replacement, from the pool of all test-set outcome prices). The added null keeps the strategy's selected matches, bets and odds fixed,
and simulates each match's result once per iteration from the market consensus probabilities, so bets in the same match stay linked. It assumes those probabilities are a reasonable basis for simulation.
""")
code("""
def strat_null(mask, odds):
    # same random stream and sampling rule as perm_pvalue(..., stratified=True) in Section 10
    rng = np.random.default_rng(SEED)
    flat_won, flat_odds = won.ravel(), odds.ravel(); ok = np.isfinite(flat_odds); per_k = mask.sum(axis=0)
    pools = [np.where(ok & (np.arange(flat_odds.size) % 3 == k))[0] for k in range(3)]
    hits = np.empty(N_PERM)
    for i in range(N_PERM):
        pick = np.concatenate([rng.choice(pools[k], per_k[k], replace=False) for k in range(3) if per_k[k]])
        hits[i] = (np.where(flat_won[pick], flat_odds[pick] - 1, -1.0)).mean()
    return hits
PM = P_MKT[te] / P_MKT[te].sum(1, keepdims=True)
def fixed_bets_null(mask, odds, n_sim=N_PERM):
    rg = np.random.default_rng(SEED + 3)
    cum = PM.cumsum(1); rows_ = np.where(mask.any(1))[0]
    m, o = mask[rows_], np.where(mask[rows_], odds[rows_], 0.0); c = cum[rows_]; stake = mask.sum()
    res = np.empty(n_sim)
    for i in range(n_sim):
        u = rg.random(len(rows_))[:, None]
        oc = (u > c).sum(1); oc = np.minimum(oc, 2)
        win = np.zeros_like(m); win[np.arange(len(rows_)), oc] = True
        res[i] = (np.where(m & win, o - 1, 0).sum() - (m & ~win).sum()) / stake
    return res
def roi_ci(mask, odds):
    pnl = np.where(mask & won, odds - 1, 0).sum(1) - (mask & ~won).sum(1); st = mask.sum(1)
    v = np.array([pnl[ix].sum() / st[ix].sum() if st[ix].sum() else np.nan for ix in BLOCK])
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5), np.nanmean(v > 0)
rows = []
for settle, odds in (("average", odds_avg), ("best", odds_max)):
    for t in (.03, .05, .07):
        mask = edge > t; r = roi(mask, odds); h = strat_null(mask, odds); f = fixed_bets_null(mask, odds); lo, hi, share = roi_ci(mask, odds)
        rows.append({"settlement": settle, "threshold": int(t*100), "bets": int(mask.sum()), "ROI": r, "ROI low (week block)": lo, "ROI high (week block)": hi, "resamples with ROI > 0": share,
                     "random null mean": h.mean(), "random null p": (np.sum(h >= r) + 1) / (N_PERM + 1), "random null 95th": np.percentile(h, 95), "random null 97.5th": np.percentile(h, 97.5),
                     "fixed-bets mean": f.mean(), "fixed-bets p": (np.sum(f >= r) + 1) / (N_PERM + 1), "fixed-bets 95th": np.percentile(f, 95)})
nulls = pd.DataFrame(rows); nulls.to_csv(R / "null_comparison.csv", index=False)
# added selection rule: edge threshold AND positive model expected return at the settlement price
rows = []
for settle, odds in (("average", odds_avg), ("best", odds_max)):
    for t, tl in ((None, "none"), (.03, 3), (.05, 5), (.07, 7)):
        mask = (P_SEL * odds > 1) if t is None else ((edge > t) & (P_SEL * odds > 1))
        r = roi(mask, odds); f = fixed_bets_null(mask, odds); h = strat_null(mask, odds); lo, hi, share = roi_ci(mask, odds)
        rows.append({"settlement": settle, "edge threshold": tl, "bets": int(mask.sum()), "home/draw/away": "/".join(str(int(v)) for v in mask.sum(0)), "hit rate": (mask & won).sum() / mask.sum(), "ROI": r, "low": lo, "high": hi,
                     "random null p": (np.sum(h >= r) + 1) / (N_PERM + 1), "fixed-bets p": (np.sum(f >= r) + 1) / (N_PERM + 1), "fixed-bets mean": f.mean()})
ev = pd.DataFrame(rows); ev.to_csv(R / "ev_filter.csv", index=False)
display(ev)
nulls
""")

md("""
## 12.12 Betting-cutoff sensitivity
Football-Data's notes say weekend odds are collected on Friday afternoons and midweek odds on Tuesday afternoons. The database stores no collection time, kick-off time or bookmaker-specific timestamp, so a collection date is **assumed** per match:
Saturday, Sunday and Monday matches use that Friday; Tuesday to Friday matches use that week's Tuesday. Only information dated strictly before the assumed collection date is treated as available.
""")
code("""
def collect_date(d):
    wd = d.weekday()
    back = {5: 1, 6: 2, 0: 3, 1: 0, 2: 1, 3: 2, 4: 3}[wd]
    return d - pd.Timedelta(days=back)
match["collect_date"] = match["date"].apply(collect_date)
hist_dates = {}
for r in match[["date", "home_team_api_id", "away_team_api_id"]].itertuples(index=False):
    hist_dates.setdefault(r.home_team_api_id, []).append(r.date); hist_dates.setdefault(r.away_team_api_id, []).append(r.date)
def prev_date(team, d):
    v = hist_dates[team]; i = bisect_left(v, d); return v[i - 1] if i else pd.NaT
ph = pd.to_datetime(pd.Series([prev_date(t, d) for t, d in zip(match["home_team_api_id"], match["date"])]))
pa_ = pd.to_datetime(pd.Series([prev_date(t, d) for t, d in zip(match["away_team_api_id"], match["date"])]))
cd = match["collect_date"].reset_index(drop=True)
late_prev = ((ph >= cd) | (pa_ >= cd)).to_numpy()
r2 = rated.merge(match[["id", "collect_date"]], on="id")
late_rating = (r2["rating_date"] >= r2["collect_date"])
wd_counts = match["date"].dt.day_name().value_counts()
cut_info = {"matches": len(match), "matches where a team's previous match is dated on or after the assumed collection date": int(late_prev.sum()),
            "of these, in the test set": int(late_prev[te].sum()),
            "player rows": len(r2), "player rows whose rating is dated on or after the assumed collection date": int(late_rating.sum()),
            "matches affected by at least one such rating": int(r2.loc[late_rating, "id"].nunique()),
            "test matches affected by at least one such rating": int(r2.loc[late_rating & r2["id"].isin(match.loc[te, "id"]), "id"].nunique()),
            "distinct rating dates in Player_Attributes": int(pa["date"].nunique())}
pd.Series(cut_info, name="value").to_frame().to_csv(R / "cutoff_counts.csv"); wd_counts.to_frame("matches").to_csv(R / "matches_by_weekday.csv")
display(wd_counts.to_frame("matches"))
pd.Series(cut_info, name="value").to_frame()
""")
code("""
# rebuild every feature as at a chosen cutoff; with cutoff = match date this must reproduce the original features
_elo_hist, _form, _h2h, _team_lineup = {}, {"home": {}, "away": {}}, {}, {}
for i, r in enumerate(match.itertuples(index=False)):
    h, a, hg, ag, d = r.home_team_api_id, r.away_team_api_id, r.home_team_goal, r.away_team_goal, r.date
    rh, ra = match["elo_home"].iat[i], match["elo_away"].iat[i]
    exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0)); s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    _elo_hist.setdefault(h, ([], []));  _elo_hist[h][0].append(d); _elo_hist[h][1].append(rh + ELO_K * (s_h - exp_h))
    _elo_hist.setdefault(a, ([], []));  _elo_hist[a][0].append(d); _elo_hist[a][1].append(ra + ELO_K * ((1 - s_h) - (1 - exp_h)))
    pts_h = 3 if hg > ag else (1 if hg == ag else 0); pts_a = 3 if ag > hg else (1 if hg == ag else 0)
    _form["home"].setdefault(h, ([], [])); _form["home"][h][0].append(d); _form["home"][h][1].append((hg, ag, pts_h))
    _form["away"].setdefault(a, ([], [])); _form["away"][a][0].append(d); _form["away"][a][1].append((ag, hg, pts_a))
    key = tuple(sorted((h, a))); _h2h.setdefault(key, ([], [])); _h2h[key][0].append(d); _h2h[key][1].append((h, hg, ag))
    for side, t in (("home", h), ("away", a)):
        pl = [getattr(r, f"{side}_player_{k}") for k in range(1, 12)]
        _team_lineup.setdefault(t, ([], [])); _team_lineup[t][0].append(d); _team_lineup[t][1].append([int(x) for x in pl if pd.notna(x)])
_pa = {}
for pid, gp in pa.groupby("player_api_id", sort=False):
    _pa[pid] = (list(gp["date"]), gp["overall_rating"].to_numpy(), gp["potential"].to_numpy())
def features_asof(cuts, rating_lag_days=0):
    out = {c: np.full(len(match), np.nan) for c in FEATURES}
    lag = pd.Timedelta(days=rating_lag_days)
    for i, (h, a, c) in enumerate(zip(match["home_team_api_id"], match["away_team_api_id"], cuts)):
        for side, t, col in (("home", h, "elo_home"), ("away", a, "elo_away")):
            ds, v = _elo_hist[t]; j = bisect_left(ds, c); out[col][i] = v[j - 1] if j else ELO_START
        for side, t in (("home", h), ("away", a)):
            if t in _form[side]:
                ds, v = _form[side][t]; j = bisect_left(ds, c)
                if j:
                    for w in (5, 10):
                        seg = v[max(0, j - w):j]
                        for q, nm in enumerate(("gf", "ga", "pts")):
                            out[f"{side}_{nm}_{w}"][i] = np.mean([x[q] for x in seg])
            ds, ln = _team_lineup[t]; j = bisect_left(ds, c)
            if j:
                ov, po = [], []
                for pid in ln[j - 1]:
                    if pid in _pa:
                        dp, o_, p_ = _pa[pid]; k = bisect_left(dp, c - lag)
                        if k: ov.append(o_[k - 1]); po.append(p_[k - 1])
                if ov:
                    out[f"{side}_squad_overall"][i] = np.nanmean(ov)
                    if np.isfinite(po).any(): out[f"{side}_squad_potential"][i] = np.nanmean(po)
        key = tuple(sorted((h, a)))
        ds, v = _h2h[key]; j = bisect_left(ds, c)
        if j:
            past = [(g1 - g2) if t == h else (g2 - g1) for t, g1, g2 in v[:j]]
            out["h2h_win_ratio"][i] = np.mean([x > 0 for x in past]); out["h2h_goal_diff"][i] = np.mean(past); out["h2h_n"][i] = j
        else:
            out["h2h_n"][i] = 0
    F = pd.DataFrame(out, columns=FEATURES)
    F["elo_diff"] = F["elo_home"] - F["elo_away"]; F["squad_overall_diff"] = F["home_squad_overall"] - F["away_squad_overall"]
    return F
F_same = features_asof(list(match["date"]))
dif = (F_same[FEATURES].to_numpy() - match[FEATURES].to_numpy().astype(float))
rebuild_ok = {"largest absolute difference from original features": float(np.nanmax(np.abs(dif))), "cells where only one version is missing": int((np.isnan(F_same[FEATURES].to_numpy()) != np.isnan(match[FEATURES].to_numpy().astype(float))).sum())}
pd.Series(rebuild_ok, name="value").to_frame().to_csv(R / "rebuild_check.csv")
pd.Series(rebuild_ok, name="value").to_frame()
""")
code("""
cuts = list(match["collect_date"])
F_cut = features_asof(cuts)
F_cut90 = features_asof(cuts, rating_lag_days=90)
SQUAD = ["home_squad_overall", "away_squad_overall", "home_squad_potential", "away_squad_potential", "squad_overall_diff"]
variants_cut = {"Original features": (X[FEATURES], FEATURES),
                "All inputs dated before assumed collection date": (F_cut, FEATURES),
                "As above, FIFA ratings lagged a further 90 days": (F_cut90, FEATURES),
                "As above, squad-strength features removed": (F_cut, [c for c in FEATURES if c not in SQUAD])}
def fit_cols(F, cols, fit_mask):
    pipe = clone(tuned["XGBoost"])
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=tscv, ensemble=True)
    cal.fit(F.loc[fit_mask, cols], y[fit_mask], clf__sample_weight=compute_sample_weight("balanced", y[fit_mask])); return cal
rows = []; CUT_PROBS = {}
for nm, (F, cols) in variants_cut.items():
    m = fit_cols(F, cols, tr); pvv, ptt = m.predict_proba(F.loc[va, cols]), m.predict_proba(F.loc[te, cols]); CUT_PROBS[nm] = ptt
    row = {"version": nm, "validation RPS": rps(y[va], pvv), "test RPS": rps(yt, ptt), "test Log-Loss": log_loss(yt, ptt, labels=[0, 1, 2]), "mean P(home) minus market": (ptt[:, 0] - P_MKT[te][:, 0]).mean()}
    ed = ptt - P_MKT[te]
    for settle, odds in (("avg", odds_avg), ("best", odds_max)):
        for t in (.03, .05, .07):
            mask = ed > t; r = roi(mask, odds); h = strat_null(mask, odds); f = fixed_bets_null(mask, odds)
            row.update({f"{settle} {int(t*100)}pp bets": int(mask.sum()), f"{settle} {int(t*100)}pp ROI": r, f"{settle} {int(t*100)}pp random p": (np.sum(h >= r) + 1) / (N_PERM + 1), f"{settle} {int(t*100)}pp fixed p": (np.sum(f >= r) + 1) / (N_PERM + 1)})
    rows.append(row)
cutoff_tab = pd.DataFrame(rows); cutoff_tab.to_csv(R / "cutoff_sensitivity.csv", index=False)
cutoff_tab.T
""")

md("## 12.13 Feature-group ablation on the validation set, correlated inputs and importance settings")
code("""
groups = {"Elo": ["elo_home", "elo_away", "elo_diff"], "Rolling form": [c for c in FEATURES if any(k in c for k in ("_gf_", "_ga_", "_pts_"))],
          "Squad strength": SQUAD, "Head-to-head": ["h2h_win_ratio", "h2h_goal_diff", "h2h_n"]}
Xdf = match[FEATURES]
base_m = fit_cols(Xdf, FEATURES, tr); base_p = base_m.predict_proba(Xdf.loc[va, FEATURES]); base_e = rps_each(y[va], base_p)
wv = V["date"].dt.to_period("W").astype(str).to_numpy(); wvs = pd.unique(wv); wvr = [np.where(wv == v)[0] for v in wvs]
rg = np.random.default_rng(SEED + 4); VB = [np.concatenate([wvr[j] for j in rg.integers(0, len(wvs), len(wvs))]) for _ in range(B)]
rows = [{"inputs": "All 23 features", "features": 23, "validation RPS": base_e.mean(), "RPS change": 0.0, "low": np.nan, "high": np.nan}]
for nm, cols in groups.items():
    keep = [c for c in FEATURES if c not in cols]
    p_ = fit_cols(Xdf, keep, tr).predict_proba(Xdf.loc[va, keep]); e_ = rps_each(y[va], p_); d = e_ - base_e
    v = [d[ix].mean() for ix in VB]
    rows.append({"inputs": f"Without {nm}", "features": len(keep), "validation RPS": e_.mean(), "RPS change": d.mean(), "low": np.percentile(v, 2.5), "high": np.percentile(v, 97.5)})
abl = pd.DataFrame(rows); abl.to_csv(R / "ablation_validation.csv", index=False)
corr = match.loc[tr, ["squad_overall_diff", "elo_diff", "home_squad_overall", "elo_home", "away_squad_overall", "elo_away"]].corr()
corr_rows = {"squad_overall_diff vs elo_diff": corr.loc["squad_overall_diff", "elo_diff"], "home_squad_overall vs elo_home": corr.loc["home_squad_overall", "elo_home"],
             "away_squad_overall vs elo_away": corr.loc["away_squad_overall", "elo_away"], "squad_overall_diff vs home_squad_overall": corr.loc["squad_overall_diff", "home_squad_overall"]}
pd.Series(corr_rows, name="training-set correlation").to_frame().to_csv(R / "feature_correlation.csv")
display(pd.Series(corr_rows, name="training-set correlation").to_frame())
abl
""")

md("## 12.14 Class-specific skill and draw diagnostics (test set)")
code("""
base = np.array([(y[tr] == c).mean() for c in range(3)])
rows = []
for name, p in fc.items():
    for k, lab in enumerate(LAB):
        y01 = (yt == k).astype(float); br = np.mean((p[:, k] - y01) ** 2); ref = np.mean((base[k] - y01) ** 2)
        d = (p[:, k] - y01) ** 2 - (base[k] - y01) ** 2
        lo, hi, _ = ci(lambda ix: 1 - ((p[ix, k] - y01[ix]) ** 2).mean() / ((base[k] - y01[ix]) ** 2).mean(), BLOCK)
        rows.append({"forecast": name, "outcome": lab, "Brier": br, "Brier of training base rate": ref, "skill vs base rate": 1 - br / ref, "skill low": lo, "skill high": hi,
                     "pred sd": p[:, k].std(), "pred min": p[:, k].min(), "pred max": p[:, k].max()})
cls = pd.DataFrame(rows); cls.to_csv(R / "class_skill.csv", index=False)
draw_counts = pd.DataFrame([{"threshold": int(t*100), "home bets": int((edge[:, 0] > t).sum()), "draw bets": int((edge[:, 1] > t).sum()), "away bets": int((edge[:, 2] > t).sum())} for t in (.03, .05, .07)])
draw_counts.to_csv(R / "bets_by_outcome_counts.csv", index=False)
display(draw_counts)
cls
""")

md("## 12.15 Betting and paired comparison for the revised pipelines (descriptive, post hoc)")
code("""
rows = []
for nm, p in {"Original selected (XGBoost, balanced, isotonic)": P_SEL,
              **{f"{r['model']}, {r['weights']}, {r['calibration']}": rev_models[(r["model"], r["weights"] == "balanced", r["calibration"])][2] for _, r in best_by_algo.iterrows()}}.items():
    d = rps_each(yt, p) - rps_m["Market consensus"]; lo, hi, v = ci(lambda ix: d[ix].mean(), BLOCK)
    row = {"pipeline": nm, "test RPS": rps(yt, p), "minus market": d.mean(), "low": lo, "high": hi, "resamples beating market": int((v < 0).sum())}
    ed = p - P_MKT[te]
    for t in (.03, .05, .07):
        mask = ed > t; row[f"bets {int(t*100)}pp"] = int(mask.sum())
        if mask.sum():
            r = roi(mask, odds_avg); f = fixed_bets_null(mask, odds_avg); row[f"ROI avg {int(t*100)}pp"] = r; row[f"fixed p {int(t*100)}pp"] = (np.sum(f >= r) + 1) / (N_PERM + 1)
    rows.append(row)
rev_bet = pd.DataFrame(rows); rev_bet.to_csv(R / "revised_betting.csv", index=False)
pd.Series({"review section seconds": time.perf_counter() - _review_start}).to_frame("value").to_csv(R / "review_runtime.csv")
rev_bet.T
""")

md("""
# 13. Second-review checks (added after the second round of supervisor feedback, November 2026)

These checks were added after the test results had been seen. Sections 1 to 12 are unchanged. The results are sensitivity analyses, not confirmatory tests.
""")
md("## 13.1 Identity of the database file")
code("""
import hashlib
_review2_start = time.perf_counter()
R2 = OUT / "review2"; R2.mkdir(exist_ok=True)
_h = hashlib.sha256()
with open(DB_PATH, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        _h.update(chunk)
db_id = pd.Series({"file": "database.sqlite", "size in bytes": DB_PATH.stat().st_size, "sha256": _h.hexdigest()}, name="value").to_frame()
db_id.to_csv(R2 / "database_identity.csv")
db_id
""")
md("## 13.2 Odds and line-up coverage of all leagues in the database")
code("""
_all = pd.read_sql("SELECT * FROM Match", con).merge(leagues, on="league_id")
_complete = sum(_all[[b + "H", b + "D", b + "A"]].notna().all(axis=1).astype(int) for b in BOOKS)
_pl = [f"{s}_player_{k}" for s in ("home", "away") for k in range(1, 12)]
league_cov = pd.DataFrame({"country": _all["country"], "league": _all["league"], "matches": 1, "at least 3 complete bookmakers": (_complete >= 3).astype(float),
                           "all 22 players listed": _all[_pl].notna().all(axis=1).astype(float)}).groupby(["country", "league"]).agg(
    {"matches": "sum", "at least 3 complete bookmakers": "mean", "all 22 players listed": "mean"}).reset_index().sort_values("matches", ascending=False)
league_cov.to_csv(R2 / "league_coverage.csv", index=False)
league_cov
""")
md("## 13.3 Tied FIFA rating rows: are the values different, and does a repeatable rule change the results?")
code("""
tie = pa.groupby(["player_api_id", "date"]).agg(rows=("overall_rating", "size"), ov_min=("overall_rating", "min"), ov_max=("overall_rating", "max"),
                                                po_min=("potential", "min"), po_max=("potential", "max"))
tie = tie[tie["rows"] > 1]
# repeatable rule: tied rows for a player and date are averaged, which does not depend on row order
pa_rule = pa.groupby(["player_api_id", "date"], as_index=False)[["overall_rating", "potential"]].mean().sort_values("date")
rated_rule = pd.merge_asof(prev_lineups.sort_values("date"), pa_rule.rename(columns={"date": "rating_date"}), left_on="date", right_on="rating_date",
                           by="player_api_id", direction="backward", allow_exact_matches=False)
squad_rule = rated_rule.groupby(["id", "team"]).agg(squad_overall=("overall_rating", "mean"), squad_potential=("potential", "mean")).reset_index()
F_rule = match[FEATURES].astype(float).reset_index(drop=True).copy()
for side in ("home", "away"):
    s_ = match[["id", f"{side}_team_api_id"]].merge(squad_rule, left_on=["id", f"{side}_team_api_id"], right_on=["id", "team"], how="left")
    F_rule[f"{side}_squad_overall"] = s_["squad_overall"].to_numpy(); F_rule[f"{side}_squad_potential"] = s_["squad_potential"].to_numpy()
F_rule["squad_overall_diff"] = F_rule["home_squad_overall"] - F_rule["away_squad_overall"]
_d = (F_rule[SQUAD].to_numpy() - match[SQUAD].to_numpy().astype(float))
_chg = np.nan_to_num(np.abs(_d)) > 1e-12
_used = rated_rule.merge(tie.reset_index()[["player_api_id", "date"]].rename(columns={"date": "rating_date"}), on=["player_api_id", "rating_date"])
m_rule = fit_cols(F_rule, FEATURES, tr); p_rule = m_rule.predict_proba(F_rule.loc[te, FEATURES]); ed_rule = p_rule - P_MKT[te]
tie_info = {"rating rows after dropping missing overall": len(pa), "player-date pairs with more than one row": len(tie), "rows in those pairs": int(tie["rows"].sum()),
            "pairs whose overall ratings differ": int((tie["ov_min"] != tie["ov_max"]).sum()), "pairs whose potential ratings differ": int((tie["po_min"] != tie["po_max"]).sum()),
            "largest overall difference within a pair": float((tie["ov_max"] - tie["ov_min"]).max()), "largest potential difference within a pair": float((tie["po_max"] - tie["po_min"]).max()),
            "line-up player rows that use a tied rating": len(_used), "matches with a changed squad feature": int(_chg.any(axis=1).sum()),
            "test matches with a changed squad feature": int(_chg[te].any(axis=1).sum()), "largest change in a squad feature": float(np.nanmax(np.abs(_d))),
            "selected configuration test RPS, original": rps(yt, P_SEL), "selected configuration test RPS, tied rows averaged": rps(yt, p_rule)}
rows = []
for settle, odds in (("average", odds_avg), ("best", odds_max)):
    for t in (.03, .05, .07):
        mask = ed_rule > t; r = roi(mask, odds); h = strat_null(mask, odds); f = fixed_bets_null(mask, odds)
        rows.append({"settlement": settle, "threshold": int(t*100), "bets": int(mask.sum()), "ROI": r, "ROI original": roi(edge > t, odds),
                     "bets also in original": int((mask & (edge > t)).sum()), "stratified p": (np.sum(h >= r) + 1) / (N_PERM + 1), "fixed-bets p": (np.sum(f >= r) + 1) / (N_PERM + 1)})
tie_bet = pd.DataFrame(rows); tie_bet.to_csv(R2 / "tied_ratings_betting.csv", index=False)
pd.Series(tie_info, name="value").to_frame().to_csv(R2 / "tied_ratings.csv")
display(tie_bet)
pd.Series(tie_info, name="value").to_frame()
""")
md("## 13.4 Whole dates kept on one side of every boundary: full rerun of tuning, calibration, selection, test and betting")
code("""
_dates = match["date"].to_numpy()
_d_tv, _d_vt = match.loc[va, "date"].min(), match.loc[te, "date"].min()
tr2 = (match["date"] < _d_tv).to_numpy(); va2 = ((match["date"] >= _d_tv) & (match["date"] < _d_vt)).to_numpy(); te2 = (match["date"] >= _d_vt).to_numpy()
_dt = _dates[tr2]
whole_folds = []
_starts = [b[0] for _, b in TimeSeriesSplit(n_splits=5).split(np.arange(tr2.sum()))] + [len(_dt)]
_edges = [_dt[s] if s < len(_dt) else None for s in _starts]
for k in range(5):
    fit_ = np.where(_dt < _edges[k])[0]
    sc_ = np.where((_dt >= _edges[k]) & (_dt < _edges[k + 1]))[0] if _edges[k + 1] is not None else np.where(_dt >= _edges[k])[0]
    whole_folds.append((fit_, sc_))
fold2 = pd.DataFrame([{"fold": k + 1, "fit matches": len(a), "fit last": str(_dt[a].max())[:10], "scored matches": len(b), "scored first": str(_dt[b].min())[:10], "scored last": str(_dt[b].max())[:10],
                       "dates on both sides": len(set(_dt[a]) & set(_dt[b]))} for k, (a, b) in enumerate(whole_folds)])
fold2.to_csv(R2 / "whole_date_folds.csv", index=False)
split2 = pd.DataFrame([{"partition": nm, "matches": int(m.sum()), "first": str(match.loc[m, "date"].min())[:10], "last": str(match.loc[m, "date"].max())[:10]} for nm, m in (("train", tr2), ("validation", va2), ("test", te2))])
split2.to_csv(R2 / "whole_date_split.csv", index=False)
display(split2); display(fold2)
w2 = compute_sample_weight("balanced", y[tr2]); yt2 = y[te2]; pm2 = P_MKT[te2]
rows, final2 = [], {}
for name, (pipe, grid, needs_weight) in candidates.items():
    gs = GridSearchCV(clone(pipe), grid, scoring=rps_scorer, cv=whole_folds, n_jobs=1)
    kw = {"clf__sample_weight": w2} if needs_weight else {}
    gs.fit(X[tr2], y[tr2], **kw)
    vs = {"uncalibrated": gs.best_estimator_}
    if name != "Logistic Regression":
        for method in ("sigmoid", "isotonic"):
            c_ = CalibratedClassifierCV(clone(gs.best_estimator_), method=method, cv=whole_folds, ensemble=True); c_.fit(X[tr2], y[tr2], **kw); vs[method] = c_
    best = min(vs, key=lambda v: rps(y[va2], vs[v].predict_proba(X[va2])))
    final2[name] = (best, vs[best]); pv, pt = vs[best].predict_proba(X[va2]), vs[best].predict_proba(X[te2])
    rows.append({"model": name, "best params": str({k.replace("clf__", ""): w for k, w in gs.best_params_.items()}), "version": best, "CV RPS": -gs.best_score_, "validation RPS": rps(y[va2], pv),
                 "test RPS": rps(yt2, pt), "test Log-Loss": log_loss(yt2, pt, labels=[0, 1, 2]), "test accuracy": accuracy_score(yt2, pt.argmax(1)),
                 "original pipeline on the same test matches": rps(yt2, final[name][1].predict_proba(X[te2]))})
rows.append({"model": "Market consensus", "test RPS": rps(yt2, pm2), "test Log-Loss": log_loss(yt2, pm2, labels=[0, 1, 2]), "test accuracy": accuracy_score(yt2, pm2.argmax(1)), "validation RPS": rps(y[va2], P_MKT[va2])})
whole_tab = pd.DataFrame(rows); whole_tab.to_csv(R2 / "whole_date_models.csv", index=False)
SEL2 = min(final2, key=lambda m: rps(y[va2], final2[m][1].predict_proba(X[va2])))
whole_tab
""")
code("""
T2 = match.loc[te2].reset_index(drop=True); P2 = final2[SEL2][1].predict_proba(X[te2]); edge2 = P2 - pm2
oa2, ob2 = T2[["odds_avg_H", "odds_avg_D", "odds_avg_A"]].to_numpy(), T2[["odds_max_H", "odds_max_D", "odds_max_A"]].to_numpy(); won2 = np.eye(3)[yt2].astype(bool)
_wk = T2["date"].dt.to_period("W").astype(str).to_numpy(); _wks = pd.unique(_wk); _wr = [np.where(_wk == w)[0] for w in _wks]
_rg = np.random.default_rng(SEED + 1); BLOCK2 = [np.concatenate([_wr[j] for j in _rg.integers(0, len(_wks), len(_wks))]) for _ in range(B)]
_diff = rps_each(yt2, P2) - rps_each(yt2, pm2); _lo, _hi, _v = ci(lambda ix: _diff[ix].mean(), BLOCK2)
_pmn = pm2 / pm2.sum(1, keepdims=True)
def roi2(mask, odds):
    return (np.where(mask & won2, odds - 1, 0).sum() - (mask & ~won2).sum()) / mask.sum()
def strat2(mask, odds):
    rg = np.random.default_rng(SEED); fw, fo = won2.ravel(), odds.ravel(); ok = np.isfinite(fo); per = mask.sum(axis=0)
    pools = [np.where(ok & (np.arange(fo.size) % 3 == k))[0] for k in range(3)]; out = np.empty(N_PERM)
    for i in range(N_PERM):
        pick = np.concatenate([rg.choice(pools[k], per[k], replace=False) for k in range(3) if per[k]]); out[i] = np.where(fw[pick], fo[pick] - 1, -1.0).mean()
    return out
def fixed2(mask, odds):
    rg = np.random.default_rng(SEED + 3); cum = _pmn.cumsum(1); r_ = np.where(mask.any(1))[0]; m, o, c = mask[r_], np.where(mask[r_], odds[r_], 0.0), cum[r_]; st = mask.sum(); out = np.empty(N_PERM)
    for i in range(N_PERM):
        oc = np.minimum((rg.random(len(r_))[:, None] > c).sum(1), 2); w_ = np.zeros_like(m); w_[np.arange(len(r_)), oc] = True
        out[i] = (np.where(m & w_, o - 1, 0).sum() - (m & ~w_).sum()) / st
    return out
rows = []
for settle, odds in (("average", oa2), ("best", ob2)):
    for t in (.03, .05, .07):
        mask = edge2 > t; r = roi2(mask, odds); h = strat2(mask, odds); f = fixed2(mask, odds)
        pnl = np.where(mask & won2, odds - 1, 0).sum(1) - (mask & ~won2).sum(1); st = mask.sum(1); v_ = np.array([pnl[ix].sum() / st[ix].sum() for ix in BLOCK2])
        rows.append({"settlement": settle, "threshold": int(t*100), "bets": int(mask.sum()), "ROI": r, "ROI low": np.percentile(v_, 2.5), "ROI high": np.percentile(v_, 97.5),
                     "stratified p": (np.sum(h >= r) + 1) / (N_PERM + 1), "fixed-bets p": (np.sum(f >= r) + 1) / (N_PERM + 1)})
whole_bet = pd.DataFrame(rows); whole_bet.to_csv(R2 / "whole_date_betting.csv", index=False)
pd.Series({"selected": f"{SEL2} ({final2[SEL2][0]})", "model minus market RPS": _diff.mean(), "low": _lo, "high": _hi, "resamples favouring model": int((_v < 0).sum()), "mean P(home) minus market": edge2[:, 0].mean()}, name="value").to_frame().to_csv(R2 / "whole_date_market.csv")
display(pd.read_csv(R2 / "whole_date_market.csv"))
whole_bet
""")
md("## 13.5 How reliable was the model on the bets it selected?")
code("""
rows = []
for settle, odds in (("average", odds_avg), ("best", odds_max)):
    for t in (.03, .05, .07):
        full = edge > t
        for lab, keep in [("All", np.ones(3, bool))] + [(LAB[k], np.eye(3, dtype=bool)[k]) for k in range(3)]:
            mask = full & keep; n = mask.sum(1); nw = (mask & won).sum(1); sm = (P_SEL * mask).sum(1); sk = (P_MKT[te] * mask).sum(1)
            lo1, hi1, _ = ci(lambda ix: (sm[ix].sum() - nw[ix].sum()) / n[ix].sum() * 100, BLOCK)
            lo2, hi2, _ = ci(lambda ix: (sk[ix].sum() - nw[ix].sum()) / n[ix].sum() * 100, BLOCK)
            rows.append({"settlement": settle, "threshold": int(t*100), "outcome": lab, "bets": int(mask.sum()), "mean model probability": P_SEL[mask].mean(), "mean market probability": P_MKT[te][mask].mean(),
                         "actual win rate": won[mask].mean(), "mean odds": odds[mask].mean(), "model expected ROI": (P_SEL[mask] * odds[mask]).mean() - 1, "market expected ROI": (P_MKT[te][mask] * odds[mask]).mean() - 1,
                         "realised ROI": roi(mask, odds), "model minus actual, points": (P_SEL[mask].mean() - won[mask].mean()) * 100, "model low": lo1, "model high": hi1,
                         "market minus actual, points": (P_MKT[te][mask].mean() - won[mask].mean()) * 100, "market low": lo2, "market high": hi2})
sel_rel = pd.DataFrame(rows); sel_rel.to_csv(R2 / "selected_bet_reliability.csv", index=False)
sel_rel
""")
md("## 13.6 Do the same bets survive a change in the information cutoff?")
code("""
rows = []
for nm, pc in CUT_PROBS.items():
    if nm == "Original features":
        continue
    ed_c = pc - P_MKT[te]
    for t in (.03, .05, .07):
        a, b = edge > t, ed_c > t
        rows.append({"version": nm, "threshold": int(t*100), "original bets": int(a.sum()), "version bets": int(b.sum()), "bets in both": int((a & b).sum()), "share of original bets kept": (a & b).sum() / a.sum(),
                     "correlation of edges": np.corrcoef(edge.ravel(), ed_c.ravel())[0, 1], "test RPS": rps(yt, pc)})
cut_stab = pd.DataFrame(rows); cut_stab.to_csv(R2 / "cutoff_bet_stability.csv", index=False)
cut_stab
""")
md("## 13.7 Separating calibration from ensembling and training history")
code("""
rows = []
for name in ("Random Forest", "XGBoost"):
    cal_ = final[name][1]; pairs_ = cal_.calibrated_classifiers_
    single = tuned[name]
    for lab_, mask_, y_ in (("validation", va, y[va]), ("test", te, y[te])):
        Xm = X[mask_]
        ens = np.mean([c_.estimator.predict_proba(Xm) for c_ in pairs_], axis=0)
        rows.append({"model": name, "set": lab_, "single uncalibrated pipeline, all training matches": rps(y_, single.predict_proba(Xm)), "average of the five fold pipelines, uncalibrated": rps(y_, ens), "average of the five fold pipelines, calibrated": rps(y_, cal_.predict_proba(Xm))})
cal_parts = pd.DataFrame(rows); cal_parts.to_csv(R2 / "calibration_decomposition.csv", index=False)
pd.Series({"second-review section seconds": time.perf_counter() - _review2_start}).to_frame("value").to_csv(R2 / "review2_runtime.csv")
cal_parts
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "RPDA8412_Market_Efficiency.ipynb")
