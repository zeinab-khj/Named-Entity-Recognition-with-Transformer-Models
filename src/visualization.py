import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from config import REPORT_DIR


REPORT_DIR = Path("/drive/MyDrive/Start_LLM/Report_Result")

PLOT_DIR = REPORT_DIR / "plots"
PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



class_df = report_df.iloc[:-3]

plt.figure(figsize=(8,5))

plt.hist(
    class_df["f1-score"],
    bins=20
)

plt.xlabel("F1 Score")
plt.ylabel("Number of Classes")
plt.title("Distribution of F1 Score Across Classes")


plt.savefig(
    PLOT_DIR / "f1_distribution.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------
# Top Classes
# -----------------------

top_classes = (
    class_df
    .sort_values(
        "f1-score",
        ascending=False
    )
    .head(10)
)


plt.figure(figsize=(8,5))

plt.barh(
    top_classes.index,
    top_classes["f1-score"]
)

plt.xlabel("F1 Score")
plt.title("Top 10 Performing Classes")
plt.gca().invert_yaxis()

plt.savefig(
    PLOT_DIR / "top_classes.png",
    bbox_inches="tight"
)
plt.close()

# -----------------------
# Worst Classes
# -----------------------
worst_classes = (
    class_df
    .sort_values(
        "f1-score",
        ascending=False
    )
    .tail(10)
)


plt.figure(figsize=(8,5))

plt.barh(
    top_classes.index,
    top_classes["f1-score"]
)

plt.xlabel("F1 Score")
plt.title("worst 10 Performing Classes")
plt.gca().invert_yaxis()

plt.savefig(
    PLOT_DIR / "worst_classes.png",
    bbox_inches="tight"
)
plt.close()

# -----------------------
# Class Support
# -----------------------
support = (
    class_df
    .sort_values(
        "support",
        ascending=False
    )
    .head(20)
)


plt.figure(figsize=(10,6))

plt.barh(
    support.index,
    support["support"]
)

plt.xlabel("Number of Samples")
plt.title("Class Distribution (Top 20)")


plt.gca().invert_yaxis()

plt.savefig(
    PLOT_DIR / "class_support.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------
# Top Confusions
# -----------------------

confusions = []


for i in range(len(cm)):

    for j in range(len(cm)):

        if i != j:

            confusions.append(
                (
                    i,
                    j,
                    cm[i,j]
                )
            )


confusions = sorted(
    confusions,
    key=lambda x:x[2],
    reverse=True
)[:10]


# Get the unique labels in the order they appear in the confusion matrix
# The confusion matrix uses the sorted unique values from the input labels
# which corresponds to the index of report_df
labels = report_df.index.tolist()


confusion_df = pd.DataFrame(
    confusions,
    columns=[
        "true",
        "predicted",
        "count"
    ]
)


confusion_df["true"] = confusion_df["true"].apply(
    lambda x: labels[x] if isinstance(x,int) else x
)

confusion_df["predicted"] = confusion_df["predicted"].apply(
    lambda x: labels[x] if isinstance(x,int) else x
)


plt.figure(figsize=(10,6))

plt.barh(
    confusion_df["true"] + " → " + confusion_df["predicted"],
    confusion_df["count"]
)


plt.xlabel("Number of Errors")
plt.title("Top Confused Class Pairs")

plt.gca().invert_yaxis()

plt.savefig(
    PLOT_DIR / "top_confusions.png",
    bbox_inches="tight"
)

plt.close()


print("Visualization completed.")
