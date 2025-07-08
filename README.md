# 📘 ReviewSummariser GPT

ReviewSummariser GPT is a fine-tuned T5-small model that takes long, detailed product reviews and generates short, helpful summaries. Three configurations (Config1, Config2, Config3) were trained and compared for performance optimization.

---

## 🚀 Project Structure
.
├── data/ # Contains ReviewLarge.csv
├── training/ # Scripts for Config1, Config2, Config3
├── evaluation/ # Evaluation scripts and outputs
├── output.png # Metrics comparison chart
├── assignment.ipynb # Full Kaggle-compatible training + evaluation notebook
├── requirements.txt
└── README.md


---

## 📊 Dataset

**File:** `ReviewLarge.csv`  
**Columns:**  
- `review`: Raw product review  
- `summary`: Ground truth summary

**Preprocessing:**

import pandas as pd

df = pd.read_csv("ReviewLarge.csv")
df = df.rename(columns={"review": "input", "summary": "target"})


---

## ⚙️ Setup Instructions

### Step 1: Install Dependencies


pip install -r requirements.txt
Required packages include:

transformers

datasets

evaluate

gradio

pandas

matplotlib

scikit-learn

tqdm

torch (CUDA recommended)

## Dataset

We use a custom ReviewLarge.csv with 1000+ samples. It includes two columns:

review — the input product review

summary — the target summary



## Preprocessing
### Step 1: Load and Clean Data

import pandas as pd
from datasets import Dataset

df = pd.read_csv("ReviewLarge.csv")
df = df.rename(columns={"review": "input_text", "summary": "target_text"})
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)



### Step 2: Tokenization

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess(example):
    model_input = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)
    label = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=60)
    model_input["labels"] = label["input_ids"]
    return model_input

tokenized_data = dataset.map(preprocess, batched=True)


## Configurations for Fine-Tuning
Config 1
Epochs: 5

Learning rate: 5e-5


from transformers import TrainingArguments

TrainingArguments(
    output_dir="./results_config1",
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    save_strategy="epoch",
    report_to="none"
)


Config 2
Epochs: 4

Learning rate: 3e-5

Modified generation: beam search, length penalty

Config 3
Epochs: 3

Learning rate: 2e-5

Lightweight version for faster training

## 🏃‍♂️ Training

from transformers import Trainer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./results_configX/checkpoint-final")
Replace X with 1, 2, or 3 depending on configuration.

## 🎯 Sentiment Classification
We use Hugging Face sentiment pipeline for extra layer of interpretation:

from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
result = sentiment("I love the battery life but hate the design.")


## 🧪 Evaluation
Metrics Used
ROUGE (rouge1, rouge2, rougeL, rougeLsum)

BLEU

BERTScore

import evaluate
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
Example Result (on 100 test samples):
Config	ROUGE-1	ROUGE-2	ROUGE-L	BERTScore
Base	0.086	0.024	0.078	-
Config 1	0.184	0.082	0.176	✅
Config 2	0.191	0.085	0.182	✅
Config 3	0.172	0.078	0.165	✅
