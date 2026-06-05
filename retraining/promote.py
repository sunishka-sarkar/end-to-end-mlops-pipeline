import json
import shutil

with open(
    "training/champion/champion_metrics.json"
) as f:

    champion = json.load(f)

with open(
    "training/models/challenger_metrics.json"
) as f:

    challenger = json.load(f)

if challenger["roc_auc"] > champion["roc_auc"]:

    shutil.copy(
        "training/models/challenger_model.pkl",
        "training/champion/champion_model.pkl"
    )

    shutil.copy(
        "training/models/challenger_metrics.json",
        "training/champion/champion_metrics.json"
    )

    print(
        "New champion promoted!"
    )

else:

    print(
        "Champion retained."
    )