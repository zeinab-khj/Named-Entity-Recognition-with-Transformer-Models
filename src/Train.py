import torch

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer
)



from config import (
    MODEL_NAME,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    NUM_LABELS,
    MODEL_DIR,
    BATCH_SIZE,
    NUM_EPOCHS
)

from metrics import compute_metrics
from datasets import load_from_disk


# -------------------------#
# Load datasets
# -------------------------#

train_dataset = load_from_disk(TRAIN_DATA_PATH)

valid_dataset = load_from_disk(VALID_DATA_PATH)



# -------------------------#
# Load final model
# -------------------------#

model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=len(id_to_label),
    id2label=id_to_label,
    label2id=label_to_id,
    torch_dtype=torch.float32
)


# -------------------------#
# Training configuration   #
# -------------------------#
training_args = TrainingArguments(
    output_dir="./ner_model_results",

    
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,



    num_train_epochs=3,

    weight_decay=0.01,

    metric_for_best_model="f1",
    

    report_to="none"
    
)

# -------------------------#
# Trainer                  #
# -------------------------#
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],   
    eval_dataset=tokenized_dataset["validation"], 
    data_collator=data_collator,    
    compute_metrics=compute_metrics,
)

# ------------------------#
#         Train           #
# ------------------------#
trainer.train()


# ------------------------#
#    Save final model     #
# ------------------------#

trainer.save_model(
    MODEL_DIR
)


print(
    f"Model saved to {MODEL_DIR}"
)
