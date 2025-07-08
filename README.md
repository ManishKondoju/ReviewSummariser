
# 🧠 Review Summarizer GPT (Config1)

This project demonstrates fine-tuning a T5-small transformer model to summarize product reviews and classify their sentiment. Built using Hugging Face Transformers, trained on a custom review-summary dataset, and deployed with Gradio UI on Hugging Face Spaces.

## 🚀 Features
- Fine-tuned T5-small for summarizing user reviews.
- Sentiment classification (positive/negative) using Hugging Face sentiment pipeline.
- Config1 trained with:
  - 5 epochs
  - Learning rate: 5e-5
  - Beam search for generation
- CPU/GPU-compatible training.
- Deployed on Hugging Face Space with UI.

## 🗃️ Dataset
- Used `ReviewLarge.csv` (1000+ product reviews).
- Columns:
  - `review` → Raw input text
  - `summary` → Target summary text
- Split: 80% train / 20% test

## 🧼 Preprocessing & Tokenization

```python
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv("ReviewLarge.csv").rename(columns={"review": "input", "summary": "target"})
dataset = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=60)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_data = dataset.train_test_split(test_size=0.2, seed=42).map(preprocess_function, batched=True)
```

## 🧠 Fine-Tuning (Config1)

```python
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./results_config1_kaggle",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer
)

trainer.train()
```

## 📈 Evaluation Metrics (on 100 test samples)

| Metric    | T5-small | Config1 |
|-----------|----------|---------|
| ROUGE-1   | 0.0860   | 0.1841  |
| ROUGE-2   | 0.0238   | 0.0816  |
| ROUGE-L   | 0.0781   | 0.1762  |
| ROUGE-Lsum| 0.0791   | 0.1769  |

## 🎯 Inference Example (Gradio)

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("summarization", model="Manish014/review-summariser-gpt-config1")

def summarize(text):
    return pipe(text)[0]["summary_text"]

gr.Interface(fn=summarize, inputs="textbox", outputs="textbox").launch()
```

## 📦 Requirements

```bash
pip install transformers datasets evaluate gradio
```

## 📁 Files

- `train_config1.py` – Fine-tuning script
- `evaluate_config1.py` – Metric-based evaluation
- `app.py` – Gradio UI
- `ReviewLarge.csv` – Dataset

---

## 👨‍💻 Author
Manish Kumar Kondoju  
📧 kondoju.m@northeastern.edu  
🔗 [LinkedIn](https://www.linkedin.com/in/manishkumar-kondoju/)  
🔗 [GitHub Repo](https://github.com/ManishKondoju/ReviewSummariser)
