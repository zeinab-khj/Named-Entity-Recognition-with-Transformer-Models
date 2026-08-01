import json
import numpy as np
import pandas as pd

from pathlib import Path

from datasets import load_from_disk
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer
)

from config import (
    MODEL_DIR,
    VALID_DATA_PATH,
    REPORT_DIR
)


def main():
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

  
    # -----------------------
    # Load model/tokenizer
    # -----------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )


    # -----------------------
    # Load validation dataset
    # -----------------------

    valid_dataset = load_from_disk(
        VALID_DATA_PATH
    )


    # -----------------------
    # Data Collator
    # -----------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )


    # -----------------------
    # Trainer
    # -----------------------

    trainer = Trainer(
        model=model,
        data_collator=data_collator
    )

    #-------------------------
    # Prediction
    #-------------------------
    predictions_output = trainer.predict(
    tokenized_valid_dataset
    )
    
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    
    predicted_labels = np.argmax(
        logits,
        axis=2 
    )


    # -----------------------
    # Metrics
    # -----------------------
    metrics = {
    "precision": results["overall_precision"],
    "recall": results["overall_recall"],
    "f1": results["overall_f1"],
    "accuracy": results["overall_accuracy"], 
    }

    with open(
        REPORT_DIR / "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

  
    # -----------------------
    # Classification Report
    # -----------------------
  
    flattened_true_labels = [item for sublist in true_labels for item in sublist]
    flattened_predicted_labels = [item for sublist in true_predictions for item in sublist]
    
    report = classification_report(
        flattened_true_labels,
        flattened_predicted_labels,
        output_dict=True,
        zero_division=0
    )
    
    
    report_df = pd.DataFrame(
        report
    ).transpose()
  
    report_df.to_csv(
    REPORT_DIR / "classification_report.csv"
    )

    # -----------------------
    # Confusion Matrix
    # -----------------------
    cm = confusion_matrix(
    flattened_true_labels,
    flattened_predicted_labels
     )

    np.save(
        REPORT_DIR / "confusion_matrix.npy",
        cm
    )


    print("Evaluation completed.")
    print(metrics)



if __name__ == "__main__":
    main()
