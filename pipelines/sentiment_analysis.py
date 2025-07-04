import torch
import torch.nn.functional as F
from pipelines.data_cleaning import clean_review_sentiment

# Pipeline for Predict Sentiment 
def predict_sentiment(review_text, tokenizer, model):
    text = clean_review_sentiment(review_text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = torch.max(probs).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[pred_class], confidence