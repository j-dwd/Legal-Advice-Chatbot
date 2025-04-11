from transformers import pipeline

summarizer = pipeline("summarization", model="nlpaueb/legal-bert-base-uncased")

def summarize_clauses(text):
    summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
    return summary[0]["summary_text"]