from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset as HfDataset
from sklearn.model_selection import train_test_split
import collections
from config import RAW_DATA_PATH, TRAIN_DATA_PATH, VALID_DATA_PATH, MODEL_NAME
    


def read_conll_2003(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith("-DOCSTART-"):
                continue

            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                splits = line.split()
                if len(splits) >= 4:
                    word = splits[0]        
                    pos_tag = splits[1]     
                    chunk_tag = splits[2]   
                    ner_tag = splits[3]     
                    current_sentence.append((word, pos_tag, chunk_tag, ner_tag))
        if current_sentence:
            sentences.append(current_sentence)

    return sentences

###-------------------------------------------------------###
###----- Parse RAW DataSet and Built it as a DataDict-----###
###-------------------------------------------------------###
file_path = RAW_DATA_PATH
dataset = read_conll_2003(file_path)
flat_data = []
for sentence_id, sentence in enumerate(dataset):
    for word, pos, chunk, ner in sentence:
        flat_data.append([sentence_id, word, pos, chunk, ner])

df = pd.DataFrame(flat_data, columns=['Sentence_ID', 'Word', 'POS', 'Chunk', 'NER_Tag'])
grouped = df.groupby("Sentence_ID").agg({
    "Word": list,
    "NER_Tag": list
}).reset_index()
grouped["Full_Sentence"] = grouped["Word"].apply(lambda tokens: " ".join(tokens))

Dataset = grouped[["Sentence_ID", "Full_Sentence", "Word", "NER_Tag"]]
train_dataset = HfDataset.from_pandas(grouped)

dataset_split = train_dataset.train_test_split(test_size=0.2, seed=42)
dataset_split["validation"] = dataset_split["test"]
del dataset_split["test"]

train = dataset_split["train"]
valid = dataset_split["validation"]


all_ner_tags = [tag for sublist in train_dataset['NER_Tag'] for tag in sublist]
ner_tag_counts = collections.Counter(all_ner_tags)
ner_tag_counts_df = pd.DataFrame(ner_tag_counts.items(), columns=['NER_Tag', 'Count']).sort_values(by='Count', ascending=False)

label_list = []
label_list = ner_tag_counts_df["NER_Tag"]


###________________________________###
###____ Tokenize & align __________###
###________________________________###

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["Word"],
        truncation=True,
        batched=False,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )

    labels = []

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:

            label_ids.append(-100)
        elif word_idx != previous_word_idx:

            text_label = examples["NER_Tag"][word_idx]
            label_ids.append(label_to_id[text_label])
        else:
           label_ids.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs



train_dataset = train.map(tokenize_and_align_labels, batched=False)
valid_dataset = valid.map(tokenize_and_align_labels, batched=False)

train_dataset.save_to_disk(TRAIN_DATA_PATH)
valid_dataset.save_to_disk(VALID_DATA_PATH)
