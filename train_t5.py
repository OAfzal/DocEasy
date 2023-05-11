import json
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


model_checkpoint = "google/flan-t5-xl"

with open("./data/data_flant5_en.json", "r") as f:
    data = json.load(f)


from torch.utils.data import Dataset
from transformers import AutoTokenizer


class FlanDataset(Dataset):
    def __init__(self, data, model_name="google/flan-t5-large"):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        prefix = example["prefix"]
        text = example["input"]
        simplified_text = example["output"]

        inputs = self.tokenizer(
            prefix + " " + text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                simplified_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        inputs["labels"] = labels["input_ids"]

        for k, v in inputs.items():
            inputs[k] = v.squeeze()

        return inputs


df = pd.DataFrame(data)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["prefix"], shuffle=True
)
val_df, test_df = train_test_split(
    val_df, test_size=0.5, random_state=42, stratify=val_df["prefix"], shuffle=True
)

train_dataset = FlanDataset(train_df.to_dict("records"))
val_dataset = FlanDataset(val_df.to_dict("records"))
test_dataset = FlanDataset(test_df.to_dict("records"))


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

batch_size = 2
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-en",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    dataloader_num_workers=8,
    gradient_accumulation_steps=8,
    fsdp=True,
)

import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train()
