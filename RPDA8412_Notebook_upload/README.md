# RPDA8412: Testing Market Efficiency in European Soccer Betting Markets Using Machine Learning

Analysis code and saved outputs for the RPDA8412 Research Report by Cameron Joshua Totham (ST10301625). Supervisor: Dr Janine Upton.

## Contents

| Path | What it is |
|---|---|
| `RPDA8412_Market_Efficiency.ipynb` | The executed notebook. Sections 1 to 11 are the original analysis. Section 12 holds the post-review checks. |
| `build_notebook.py` | Generates the notebook's cells. |
| `outputs/` | Tables (CSV) and figures (PNG) used in Chapter 4 of the report. |
| `outputs/review/` | Tables from the post-review checks. |
| `requirements.txt` | Versions used for the reported run. |

## Data

The data file is not included. Download `database.sqlite` from the European Soccer Database (Mathien, 2016):
<https://www.kaggle.com/datasets/hugomathien/soccer>

Place it at `data/database.sqlite`, or set the `SOCCER_DATA_DIR` environment variable to its folder. When no local file is found, the notebook tries to download it with `kagglehub`, which is available in Google Colab. The Kaggle page does not provide a fixed version number for the copy used here.

The database contains names and attributes of professional players. It is not redistributed here, and no individual player is named in the outputs. Bookmaker odds are never used as model inputs.

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 RPDA8412_Market_Efficiency.ipynb
```

The run used Python 3.13.5 with the random seed fixed at 42. Re-running reproduced the reported scores, and only run times differed. The original stages take a few minutes on a 16-core laptop, and Section 12 takes about 17 minutes more.

## Note on the post-review checks

Section 12 was added after supervisor feedback, when the test results had already been seen. Its results are sensitivity analyses and revisions, not confirmatory tests.

The simulation is a research test only. No real wagers were placed, and it is not betting advice.
