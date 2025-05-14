import torch
import json
import pandas as pd
import sys
import io
import re
import logging
from pydantic import BaseModel, ConfigDict, Field, field_validator

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict


logger = logging.getLogger("uvicorn.info")

from thematico.llm import (
    json_themes,
    prompt_for_themes,
    define_themes,
    generate_themes,
)

from thematico.io import (
    validate_json_format,
)

from thematico.keywords import (
    extract_keywords,
    highlight_keywords,
    match_keywords_to_codes,
    analyze_text,
)


from thematico.sentiment import predict_sentiment, split_sentiment, sentiment_analysis
from thematico.services.strategies import InductiveCoding, DeductiveCoding

from keybert import KeyBERT
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer, util
import ollama
from ollama import Client as Olama, ResponseError as OlamaCompletionError


class PipelineConfig(BaseModel):
    bert_embedder: str = Field(
        "all-mpnet-base-v2", description="The embedder model used for KeyBERT"
    )
    bert_model: str = Field(
        "KeyBERT", description="The model used for keyword extraction"
    )
    custom_prompt: Optional[str] = Field(
        None, description="Custom prompt for keyword extraction"
    )
    deductive_max_kw: int = Field(15, description="Max keywords for deductive mode")
    deductive_top_k: int = Field(30, description="Top K keywords for deductive mode")
    device: Optional[str] = Field(None, description="The device to run the model on")
    dtype: Optional[torch.dtype] = Field(
        torch.bfloat16, description="Torch dtype to use for models"
    )
    keyword_threshold: float = Field(
        0.1, description="The threshold for keyword extraction"
    )
    keyword_top_n: int = Field(20, description="The number of top keywords to extract")
    keyword_min_chars: int = Field(
        700, description="Minimum chars for keyword extraction"
    )
    keyword_max_chars: int = Field(
        1000, description="Maximum chars for keyword extraction"
    )
    llm_model: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="The model used for theme interpretation",
    )
    llm_tokenizer: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="The tokenizer used for theme interpretation",
    )
    sentence_model: str = Field(
        "all-MiniLM-L6-v2", description="The model used for clustering of the keywords."
    )
    sentiment_analysis_model: str = Field(
        "nlptown/bert-base-multilingual-uncased-sentiment",
        description="The model used for sentiment analysis.",
    )
    token: Optional[str] = Field(
        None, description="The token for Hugging Face model access"
    )
    cluster_accuracy: int = Field(30, description="The accuracy for clustering")
    min_cluster_size: int = Field(2, description="The minimum size for clustering")
    custom_codes: Optional[List[str]] = Field(
        None, description="Codes for deductive mode"
    )
    deductive_threshold: float = Field(
        0.5, description="The threshold for deductive mode"
    )
    mode: str = Field(
        "inductive", description="The mode of operation (inductive or deductive)"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ollama_url: str = Field("", description="The URL for the Ollama server")
    ollama_model: str = Field(
        "qwen2.5:32b-instruct-q4_1", description="The Ollama model name"
    )
    sent_max_len: int = Field(256, description="Max length for sentiment analysis")
    sent_threshold: float = Field(0.5, description="Threshold for sentiment analysis")
    use_ollama: bool = Field(True, description="Use Ollama for LLM")

    @field_validator("device", mode="before")
    @classmethod
    def set_device(cls, v):
        if v is not None:
            return torch.device(v)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self.models = {}

    def _load_analysis_models(self):
        # Keyword extractor
        if "bert_model" not in self.models:
            if "bert_embedder" not in self.models:
                logger.info(
                    f"Loading BERT model with embedder: {self.config.bert_embedder}"
                )
                self.models["bert_embedder"] = SentenceTransformer(
                    self.config.bert_embedder
                ).to(self.device)
                self.models["bert_model"] = KeyBERT(self.models["bert_embedder"])

        # Sentiment
        if "sentiment_model" not in self.models:
            logger.info(
                f"Loading Sentiment model: {self.config.sentiment_analysis_model}"
            )
            self.models["sentiment_model"] = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.config.sentiment_analysis_model,
                ).to(self.device)
            )
        if "sentiment_tokenizer" not in self.models:
            logger.info(
                f"Loading Sentiment tokenizer: {self.config.sentiment_analysis_model}"
            )
            self.models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained(
                self.config.sentiment_analysis_model,
                use_auth_token=self.config.token,
            )
        # Clustering + sentence embeddings
        if "sentence_model" not in self.models:
            logger.info(f"Loading Sentence model: {self.config.sentence_model}")
            self.models["sentence_model"] = SentenceTransformer(
                self.config.sentence_model
            ).to(self.device)

    def _llm_ollama(self):
        # Olamma HTTP client
        self.models["llm_client"] = Olama(host=self.config.ollama_url, timeout=120)

    def _load_llm_locally(self):
        # Load the LLM model and tokenizer
        if "llm_model" not in self.models:
            logger.info(f"Loading LLM model: {self.config.llm_model}")
            self.models["llm_model"] = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model,
                torch_dtype=(self.config.dtype if self.config.dtype else torch.float16),
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_auth_token=self.config.token,
            ).to(self.device)
        if "llm_tokenizer" not in self.models:
            logger.info(f"Loading LLM tokenizer: {self.config.llm_tokenizer}")
            self.models["llm_tokenizer"] = AutoTokenizer.from_pretrained(
                self.config.llm_tokenizer,
                use_auth_token=self.config.token,
            )

    def _load_deductive_models(self):
        # Sentiment model
        if "sentiment_model" not in self.models:
            logger.info(
                f"Loading Sentiment model: {self.config.sentiment_analysis_model}"
            )
            self.models["sentiment_model"] = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.config.sentiment_analysis_model,
                    torch_dtype=self.config.dtype,
                    low_cpu_mem_usage=True,
                    use_auth_token=self.config.token,
                ).to(self.device)
            )
        if "sentiment_tokenizer" not in self.models:
            logger.info(
                f"Loading Sentiment tokenizer: {self.config.sentiment_analysis_model}"
            )
            self.models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained(
                self.config.sentiment_analysis_model,
            )

    def load_models(self, use_ollama: bool):
        self._load_analysis_models()

        if use_ollama:
            self._llm_ollama()
        else:
            self._load_llm_locally()

    def get_models(self, use_ollama: bool):
        self._load_analysis_models()
        return self.models


def run_pipeline(
    data: Dict[str, Any],
    models,
    cfg,
    wrapper: ModelWrapper,
    skip_keyword_extraction: bool = False,
) -> Dict[str, Any]:

    required = [
        "bert_model",
        "sentence_model",
        "sentiment_model",
        "sentiment_tokenizer",
    ]
    if cfg.use_ollama:
        required.append("llm_client")
    else:
        required += ["llm_model", "llm_tokenizer"]

    missing = [model for model in required if model not in models]
    if missing:
        logger.error(f"Missing models: {missing}")

    if not validate_json_format(data):
        logger.error("Invalid JSON format.")
        raise ValueError("Invalid JSON format.")

    for entry in data["result"]:
        entry.setdefault("keywords", [])
        entry.setdefault("codes", {})

    try:
        if not skip_keyword_extraction and any(
            len(entry.get("keywords", [])) == 0 for entry in data["result"]
        ):
            data["result"], text_keywords = extract_keywords(
                data["result"],
                models["bert_model"],
                cfg.keyword_threshold,
                cfg.keyword_top_n,
                cfg.keyword_min_chars,
                cfg.keyword_max_chars,
            )
            logger.info("Keywords extracted successfully.")
        else:
            text_keywords = [
                kw for entry in data["result"] for kw in entry.get("keywords", [])
            ]
            logger.info("Keywords already present, skipping extraction.")
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        text_keywords = []

    try:
        data["result"] = sentiment_analysis(
            data["result"],
            models["sentiment_model"],
            models["sentiment_tokenizer"],
            cfg.sent_max_len,
            cfg.sent_threshold,
        )
        logger.info("Sentiment analysis completed successfully.")
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")

    similarity = {}
    clusters = {}
    combined = data

    if cfg.mode == "inductive":
        strategy = InductiveCoding(
            wrapper=wrapper,
            sentence_model=models["sentence_model"],
            use_ollama=cfg.use_ollama,
        )
        try:
            combined, similarity = strategy.code(
                data,
                text_keywords,
                cfg.cluster_accuracy,
                cfg.min_cluster_size,
            )
            logger.info("Inductive coding completed successfully with Olama.")
        except Exception as e:
            logger.error(f"Error in inductive coding: {e}")

    else:
        strategy = DeductiveCoding(models["sentence_model"])
        try:
            combined, clusters = strategy.code(
                data,
                text_keywords,
                cfg.custom_codes,
                cfg.deductive_threshold,
                cfg.deductive_top_k,
                cfg.deductive_max_kw,
            )
            logger.info("Deductive coding completed successfully.")
        except Exception as e:
            logger.error(f"Error in deductive coding: {e}")

    json_prompt_themes = json_themes()
    prompt_themes = prompt_for_themes(text_keywords, json_prompt_themes)
    if cfg.use_ollama:
        try:
            themes = generate_themes(
                prompt_themes,
                use_ollama=True,
                ollama_client=models["llm_client"],
                ollama_model_name=cfg.ollama_model,
                llm_model=models.get("llm_model"),
                llm_tokenizer=models.get("llm_tokenizer"),
                wrapper=wrapper,
                device=cfg.device,
                dtype=cfg.dtype,
            )
        except OlamaCompletionError as e:
            logger.error(f"Error with Ollama: {e}")
            themes = {}
    else:
        try:
            with torch.no_grad():
                themes = define_themes(
                    models["llm_model"],
                    models["llm_tokenizer"],
                    prompt_themes,
                    device=cfg.device,
                    dtype=cfg.dtype,
                )
        except Exception as e:
            logger.error(f"Error with local LLM: {e}")
            themes = {}

    themes_list = [list(theme.values())[0] for theme in themes.get("themes", [])]
    combined["themes"] = themes_list
    logger.info("Themes generated successfully.")

    logger.info("Pipeline completed successfully.")
    return (combined, text_keywords, themes_list, similarity, clusters)


def analyze_custom_string(
    string: str,
    combined_result: dict[str, Any],
    models: dict[str, Any],
    cfg: PipelineConfig,
) -> dict[str, Any]:
    """
    Analyze a custom user-entered string with sentiment and keyword-code analysis.

    Args:
        string (str): The custom string to analyze.
        combined_result (dict): The combined result from the pipeline.
        models (dict): The models used for analysis.
        cfg (PipelineConfig): The configuration for the pipeline.

    Returns:
        dict: The analysis result containing keywords, codes, and sentiment.
    """
    if isinstance(combined_result, list):
        combined_result = {"result": combined_result}
    try:
        matched_keywords, matched_codes = analyze_text(string, combined_result)

        custom_analysis_entry = {
            "text": string,
            "keywords": matched_keywords,
            "codes": matched_codes,
            "sentiment": None,
        }
        logger.info("Custom string analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error analyzing custom string: {e}")

        custom_analysis_entry = {
            "text": string,
            "keywords": [],
            "codes": {},
            "sentiment": [],
        }

    try:
        if not models.get("sentiment_model") or not models.get("sentiment_tokenizer"):
            logger.error("Sentiment model or tokenizer not loaded.")
        if len(string.split()) <= 225:
            label, confidence, flagged = predict_sentiment(
                string,
                models["sentiment_model"],
                models["sentiment_tokenizer"],
                max_len=cfg.sent_max_len,
                threshold=cfg.sent_threshold,
            )

            custom_analysis_entry["sentiment"] = {
                "overall_sentiment": label,
                "confidence": confidence,
                "uncertain": flagged,
            }

            logger.info("Sentiment analysis for custom string completed successfully.")

        else:
            custom_chunks = split_sentiment(string, max_len=225)
            chunk_sentiment = [
                predict_sentiment(
                    chunk,
                    models["sentiment_model"],
                    models["sentiment_tokenizer"],
                    cfg.device,
                )
                for chunk in custom_chunks
            ]
            labels = [sent[0] for sent in chunk_sentiment]
            confidences = [sent[1] for sent in chunk_sentiment]
            flags = [sent[2] for sent in chunk_sentiment]

            majority_sent = Counter(labels).most_common(1)[0][0]
            avg_conf = sum(confidences) / len(confidences)
            flagged = any(flags)

            custom_analysis_entry["sentiment"] = {
                "overall_sentiment": majority_sent,
                "confidence": avg_conf,
                "uncertain": flagged,
            }

            logger.info("Sentiment analysis for custom string completed successfully.")

    except Exception as e:
        logger.error(f"Error in sentiment analysis for custom string: {e}")
        custom_analysis_entry["sentiment"] = None

    return custom_analysis_entry
