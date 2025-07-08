---
library_name: transformers
tags:
- text2text-generation
- summarization
- product-review
- sentiment-analysis
- t5-small
- huggingface
---

# 📝 Review Summariser GPT - Config1

A fine-tuned `t5-small` model that generates concise summaries from product reviews. This model is part of the ReviewSummariserGPT project and also pairs well with a Hugging Face sentiment analysis pipeline for classifying tone.

---

## Model Details

### Model Description

This is a sequence-to-sequence Transformer model based on `t5-small`, fine-tuned on a dataset of 1,000 product review → summary pairs. It is designed to take in a review (e.g., from Amazon or Yelp) and output a short, helpful summary.

- **Developed by:** Manish Kumar Kondoju
- **Finetuned from model:** [`t5-small`](https://huggingface.co/t5-small)
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Model type:** Seq2Seq Transformer (Text-to-Text Generation)
- **Shared by:** Manish014

### Model Sources

- **Repository:** [https://huggingface.co/Manish014/review-summariser-gpt-config1](https://huggingface.co/Manish014/review-summariser-gpt-config1)
- **Demo:** Coming soon via Gradio Hugging Face Space

---

## Uses

### Direct Use

- Generate summaries for user reviews to improve content digestibility.
- Enhance e-commerce UX by auto-summarizing reviews.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Manish014/review-summariser-gpt-config1")
model = AutoModelForSeq2SeqLM.from_pretrained("Manish014/review-summariser-gpt-config1")

input_text = "summarize: The build quality is terrible and the support team was unhelpful."
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
outputs = model.generate(inputs["input_ids"], max_length=60)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
