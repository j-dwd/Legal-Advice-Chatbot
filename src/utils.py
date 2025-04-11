import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}