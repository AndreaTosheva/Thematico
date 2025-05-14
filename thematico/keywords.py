import nltk
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT
from collections import defaultdict, Counter
from keyphrase_vectorizers import KeyphraseCountVectorizer
import logging
from typing import Dict, Any, List, Optional
import numpy as np

import re

logger = logging.getLogger("uvicorn.info")


def extract_keywords(
    result: list[dict],
    model: KeyBERT,
    threshold: float,
    top_n: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[dict], list[str]]:
    """Extract keywords from the text of each entry in the result list.

    Args:
        result (list[dict]): A list of dictionaries containing text entries
        model (KeyBERT): A keyword extraction model.
        threshold (float): The minimum score for a keyword to be considered.
        top_n (int): The number of top keywords to extract from each text chunk.
        min_chars (int): Minimum character length for a text chunk.
        max_chars (int): Maximum character length for a text chunk.

    Returns:
        tuple[list[dict], list[str]]: A tuple containing the updated result list and a list of keywords.
    """

    def split_sentences(text: str, min_chars: int, max_chars: int) -> list[str]:
        """Split a text into chunks of sentences with a minimum and maximum character length.

        Args:
            text (str): The input text to split.
            min_chars (int): The minimum character length for a text chunk.
            max_chars (int): The maximum character length for a text chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")

        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if not sentence.strip():
                continue

            temp_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(temp_chunk) < min_chars:
                current_chunk = temp_chunk
                continue

            if min_chars <= len(temp_chunk) <= max_chars:
                chunks.append(temp_chunk.strip())
                current_chunk = ""
                continue

            if len(temp_chunk) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(sentence) > max_chars:
                    chunks.append(sentence.strip())
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def remove_repetitive_words(keywords: dict) -> list[str]:
        """Removes repetitive or overlapping keywords based on their scores.

        Args:
            keywords (dict): A dictionary of keywords and their associated scores.

        Returns:
            list[str]: A list of unique, non-overlapping keywords.
        """

        keywords_sorted = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        final_keywords = []

        for keyword, score in keywords_sorted:
            if not any(keyword in f or f in keyword for f in final_keywords):
                final_keywords.append(keyword)

        return final_keywords

    full_text = " ".join(entry["text"] for entry in result if entry.get("text"))
    chunks = split_sentences(full_text, min_chars, max_chars)

    all_keywords = []
    keyword_locations = defaultdict(set)
    keyword_scores = defaultdict(float)
    vectorizer = KeyphraseCountVectorizer(lowercase=True, stop_words="english")

    for chunk in chunks:
        try:
            keywords = model.extract_keywords(
                chunk,
                vectorizer=vectorizer,
                keyphrase_ngram_range=(1, 4),
                use_mmr=False,
                top_n=top_n,
            )
        except Exception as e:
            logger.warning(
                f"Error in extracting keywords from chunks: '{chunk}', error: {e}"
            )
            continue

        for keyword, score in keywords:
            keyword = keyword.lower()
            keyword_scores[keyword] = max(keyword_scores[keyword], score)
            keyword_locations[keyword].add(chunk)

    if not keyword_scores:
        logger.warning("No keywords extracted.")
        return result, []

    M = max(keyword_scores.values()) or 1
    keyword_scores = {kw: score / M for kw, score in keyword_scores.items()}
    filtered = {kw: sc for kw, sc in keyword_scores.items() if sc >= threshold}
    filtered_keywords = remove_repetitive_words(filtered)

    for entry in result:
        entry_text = entry.get("text", "").lower()
        entry_keywords = []

        for keyword in filtered_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", entry_text):
                if keyword in entry_text:
                    entry_keywords.append(keyword)

        entry["keywords"] = entry_keywords

    return result, filtered_keywords


def highlight_keywords(text, keywords):
    """Highlight the extracted keywords in the transcription text."""
    keywords = sorted(keywords, key=len, reverse=True)

    def replace_keyword(match):
        keyword = match.group(0)
        return f"<span style='background-color: yellow; font-weight: bold;'>{keyword}</span>"

    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(replace_keyword, text)
    return text


def match_keywords_to_codes(custom_keywords: list, json_groups: dict) -> dict:
    """Match keywords to codes based on the custom keywords and the JSON groups.

    Args:
        custom_keywords (list): The list of keywords extracted from the custom string passed by the user.
        json_groups (dict): The dictionary containing the JSON groups of keywords and codes.

    Returns:
        dict: A dictionary containing the matched codes and their corresponding keywords from the user passed prompt.
    """
    matched_codes = {}

    for phrase in custom_keywords:
        phrase = phrase.lower().strip()
        found_match = False

        for code, keyword_list in json_groups.items():
            if phrase in [keyword.lower().strip() for keyword in keyword_list]:
                matched_codes.setdefault(code, []).append(phrase)
                found_match = True

        if not found_match:
            words_in_phrase = set(phrase.split())

            for code, keyword_list in json_groups.items():
                keyword_set = {
                    word.lower().strip()
                    for keyword in keyword_list
                    for word in keyword.split()
                }
                matched_words = words_in_phrase.intersection(keyword_set)
                if matched_words:
                    matched_codes.setdefault(code, []).append(phrase)

    return matched_codes


def analyze_text(
    string: str, combined_result: dict[str, Any]
) -> tuple[list[str], dict[str, list[str]]]:
    """Analyze a given text by extracting relevant keywords and categorizing them under appropriate codes.

    Args:
        string (str): The text to be analyzed.
        combined_result (dict[str, Any]): The result of the keyword extraction and code definition.
    Returns:
        tuple[list[str], dict[str, list[str]]]: A tuple containing the matched keywords and codes for the passed text.
    """
    results = combined_result.get("result", [])
    # Split the input text into sentences on ., !, or ? followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", string.strip().lower())

    matched_keywords = set()
    matched_codes = defaultdict(set)

    # Process each sentence separately
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        # Exact substring match on the full sentence
        matching_segments = [
            segment
            for segment in results
            if s in segment.get("text", "").strip().lower()
        ]
        # Tokenize the sentence into individual words
        sentence_words = set(re.findall(r"\b\w+\b", s))

        # Fallback to token overlap if no exact match found
        if not matching_segments:
            for segment in results:
                text_words = set(
                    re.findall(r"\b\w+\b", segment.get("text", "").strip().lower())
                )
                if sentence_words & text_words:
                    matching_segments.append(segment)

        # From each matched segment, extract and accumulate relevant keywords
        for segment in matching_segments:
            for code, info in segment.get("codes", {}).items():
                for keyword in info.get("keywords", []):
                    keyword_words = set(re.findall(r"\b\w+\b", keyword.lower()))
                    if keyword_words & sentence_words:
                        matched_keywords.add(keyword)
                        if code not in matched_codes:
                            matched_codes[code] = set()
                        matched_codes[code].add(keyword)

    matched_keywords = list(matched_keywords)
    matched_codes = {k: list(v) for k, v in matched_codes.items()}

    return matched_keywords, matched_codes
