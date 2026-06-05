import json

with open(
    "training/champion/champion_metrics.json"
) as f:

    champion = json.load(f)

with open(
    "training/models/challenger_metrics.json"
) as f:

    challenger = json.load(f)

print("\nChampion:")
print(champion)

print("\nChallenger:")
print(challenger)

if challenger["roc_auc"] > champion["roc_auc"]:

    print(
        "\nNew model is BETTER"
    )

else:

    print(
        "\nKeep existing champion"
    )