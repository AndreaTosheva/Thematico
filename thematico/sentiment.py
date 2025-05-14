import torch
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def predict_sentiment(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    threshold: float = 0.5,
) -> tuple[str, float, bool]:
    """
    Predicts the sentiment of the given text using the provided model and tokenizer.
    Args:
        text (str): The input text to analyze.
        model (AutoModelForSequenceClassification): The pre-trained sentiment analysis model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        max_len (int): The maximum length of the tokenized input text for the model.
        threshold (float): The confidence threshold for uncertain predictions.

    Returns:
        Tuple[str, float, float, float, bool]:
        - str: Sentiment label (e.g., "Positive").
        - float: Confidence score from the model.
        - bool: Flag indicating whether confidence is below the threshold (uncertain).
    """

    sentiment_map = {
        0: "Very negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very positive",
    }

    inputs = tokenizer(
        text,
        max_length=int(max_len),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

        label_id = torch.argmax(probs).item()
        confidence = probs[label_id].item()
        sentiment_label = sentiment_map[label_id]

        flagged = confidence < threshold

    return sentiment_label, confidence, flagged


def split_sentiment(text: str, max_len: int = 225) -> list[str]:
    """Splits the text into chunks of sentences with a maximum length.

    Args:
        text (str): The input text to split.
        max_len (int): The maximum length of each chunk.

    Returns:
        list[str]: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        if len(" ".join(current_chunk + [sentence]).split()) <= int(max_len):
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def sentiment_analysis(
    data: dict,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    max_len: int = 256,
    threshold: float = 0.5,
) -> dict:
    """Performs sentiment analysis on the text data in the provided dictionary.

    Args:
        data (dict): The input data containing text entries to analyze.
        model (AutoModelForSequenceClassification): The pre-trained sentiment analysis model.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        max_len (int, optional): The length to which the paragraphs are supposed to be. Defaults to 256.
        threshold (float, optional): Value below which the score is considered "uncerain". Defaults to 0.5.

    Returns:
        dict: The input data with added sentiment analysis results.
    """

    for result in data:
        text = result.get("text", "")

        if len(text.split()) <= int(max_len):
            label, confidence, flagged = predict_sentiment(
                text, model, tokenizer, max_len, threshold
            )
            result["sentiment_score"] = {
                "overall_sentiment": label,
                "confidence": confidence,
                "uncertain": flagged,
            }

        else:
            chunks = split_sentiment(text, max_len)
            chunk_sentiments = [
                predict_sentiment(chunk, model, tokenizer, max_len, threshold)
                for chunk in chunks
            ]

            labels = [sent[0] for sent in chunk_sentiments]
            confidences = [sent[1] for sent in chunk_sentiments]
            flags = [sent[2] for sent in chunk_sentiments]

            majority_sent = Counter(labels).most_common(1)[0][0]
            avg_conf = sum(confidences) / len(confidences)
            flagged = any(flags)

            result["sentiment"] = {
                "overall_sentiment": majority_sent,
                "confidence": avg_conf,
                "uncertain": flagged,
            }

    return data
