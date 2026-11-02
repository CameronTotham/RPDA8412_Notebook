# Testing Market Efficiency in European Soccer Betting Markets Using Machine Learning

Analysis code and saved outputs for the RPDA8412 Research Report by Cameron Joshua Totham (ST10301625). Supervisor: Dr Janine Upton.

The project asks whether machine learning models trained on pre-match football data can beat the probabilities implied by bookmaker odds, and whether betting on any disagreement with the market would have made a profit. Bookmaker odds are never used as model inputs.


## Study design

| Item | Detail |
|---|---|
| Data | European Soccer Database (Mathien, 2016) |
| Leagues | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 |
| Matches | 14,575, from 9 August 2008 to 17 May 2016 |
| Split, in date order | Train 10,202, validation 2,186, test 2,187 |
| Test period | 11 April 2015 to 17 May 2016 |
| Inputs | 23 pre-match features: Elo, recent form, squad ratings, head-to-head |
| Models | Logistic Regression, Random Forest, XGBoost |
| Benchmark | Market consensus probabilities from bookmaker odds |
| Selected model | XGBoost with isotonic calibration, chosen on the validation set |

## Main results

Test-set scores, with bootstrap 95% confidence intervals. Lower RPS and log-loss are better.

| Model | RPS | Log-loss | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.2095 [0.2042, 0.2149] | 1.0264 [1.0090, 1.0426] | 0.4614 [0.4417, 0.4819] |
| Random Forest | 0.2048 [0.1993, 0.2099] | 1.0014 [0.9840, 1.0181] | 0.5048 [0.4851, 0.5249] |
| XGBoost | 0.2034 [0.1976, 0.2091] | 0.9959 [0.9770, 1.0144] | 0.5098 [0.4893, 0.5295] |
| Market consensus | 0.1988 [0.1933, 0.2041] | 0.9794 [0.9613, 0.9978] | 0.5194 [0.4989, 0.5396] |

Simulated betting on the selected model, at edge thresholds of 3, 5 and 7 percentage points (pp). Intervals are 95% calendar-week block bootstrap intervals. Each p-value is one-sided and comes from a null distribution in which bets are simulated at random.

| Settlement odds | Threshold | Bets | ROI | 95% interval | p (random betting, stratified) | p (fixed bets) |
|---|---|---|---|---|---|---|
| Average | 3 pp | 1,727 | -7.95% | -15.16% to -0.52% | 0.754 | 0.798 |
| Average | 5 pp | 1,087 | -8.89% | -18.16% to +0.74% | 0.790 | 0.806 |
| Average | 7 pp | 661 | -2.19% | -15.71% to +11.68% | 0.248 | 0.357 |
| Best | 3 pp | 1,727 | -2.66% | -10.60% to +5.40% | 0.661 | 0.822 |
| Best | 5 pp | 1,087 | -3.70% | -13.76% to +6.56% | 0.716 | 0.819 |
| Best | 7 pp | 661 | +3.74% | -10.96% to +18.71% | 0.179 | 0.377 |

The market had the lowest RPS, and no betting result was statistically significant. This does not prove the market is efficient: inefficiency was not demonstrated with these models, features and prices. The report sets out the full interpretation and limits.

## Repository contents

The working files are in the `RPDA8412_Notebook_upload` folder.

| Path | What it is |
|---|---|
| `RPDA8412_Market_Efficiency.ipynb` | The executed notebook. Sections 1 to 11 are the original analysis. Section 12 holds the post-review checks. |
| `build_notebook.py` | Generates the notebook's cells. |
| `outputs/` | Tables (CSV) and figures (PNG) from the original analysis, plus the run record. |
| `outputs/review/` | Tables from the post-review checks. |
| `requirements.txt` | Versions used for the reported run. |

The notebook in the top level of the repository is an identical copy of the one in the folder.

## Data

The data file is not included. Download `database.sqlite` from the [European Soccer Database on Kaggle](https://www.kaggle.com/datasets/hugomathien/soccer).

Place it at `RPDA8412_Notebook_upload/data/database.sqlite`, or set the `SOCCER_DATA_DIR` environment variable to its folder. When no local file is found, the notebook tries to download it with `kagglehub`, which is available in Google Colab. The Kaggle page does not provide a fixed version number for the copy used here.

The database contains names and attributes of professional players. It is not redistributed here, and no individual player is named in the outputs.

## Reproducing the results

```bash
cd RPDA8412_Notebook_upload
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 RPDA8412_Market_Efficiency.ipynb
```

The run used Python 3.13.5 with the random seed fixed at 42. Re-running reproduced the reported scores, and only run times differed. The original stages take a few minutes on a 16-core laptop, and Section 12 takes about 17 minutes more.


## The simulation is a research test only. No real wagers were placed, and it is not betting advice.

## Reference

Mathien, H. 2016. European Soccer Database [Dataset]. Kaggle. <https://www.kaggle.com/datasets/hugomathien/soccer> [Accessed: 14 October 2026].
