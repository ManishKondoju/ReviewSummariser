import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("Manish014/review-summariser-gpt-config1")
tokenizer = AutoTokenizer.from_pretrained("Manish014/review-summariser-gpt-config1")
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to summarize + classify
def summarize_and_classify(review):
    if not review.strip():
        return "Please enter a review.", "N/A"
    inputs = tokenizer("summarize: " + review, return_tensors="pt", truncation=True)
    output_ids = model.generate(inputs["input_ids"], max_length=60, min_length=10, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    sentiment = sentiment_pipeline(review)[0]['label']
    return summary, sentiment

# Gradio Interface
iface = gr.Interface(
    fn=summarize_and_classify,
    inputs=gr.Textbox(label="📝 Enter a Product Review", lines=4, placeholder="Paste a review here..."),
    outputs=[
        gr.Textbox(label="📌 Generated Summary"),
        gr.Textbox(label="💬 Sentiment")
    ],
    title="🧠 Review Summariser GPT + Sentiment Classifier",
    description="Paste a product review to generate a short summary and detect sentiment using a fine-tuned T5 model.",
    examples=[
        ["This is hands down the best vacuum cleaner I’ve ever owned. It’s lightweight, powerful, and the battery lasts forever!"],
        ["Product arrived broken and late. Extremely disappointed with the quality and packaging."],
        ["Good value for the price. The headphones sound great, but the build feels a bit cheap."]
    ]
)

iface.launch()
