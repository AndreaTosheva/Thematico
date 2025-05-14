from typing import Protocol, List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from thematico.services.pipeline import ModelWrapper
import ollama
from ollama import Client as Olama, ResponseError as OlamaCompletionError

import torch
from thematico.clustering import (
    cluster_keywords,
    cluster_custom_codes,
    rename_clusters,
    calculate_default_code_similarity,
)
from thematico.llm import (
    json_codes,
    prompt_for_codes,
    define_codes,
    assign_codes_to_data,
    generate_codes,
)


class CodingStrategy(Protocol):
    def code(self, data, keywords, **kwargs) -> Dict[str, Any]:
        """
        Common interface: should return a dict with at least:
          - 'entries': the list of annotated entries
          - any other artifacts (clusters, raw codes, similarity, etc.)
        """


class InductiveCoding:

    def __init__(
        self,
        wrapper: "ModelWrapper",
        sentence_model,
        use_ollama: bool = True,
    ):
        self.sentence_model = sentence_model
        self.wrapper = wrapper
        self.use_ollama = use_ollama

    def code(self, data, keywords, cluster_acc, min_size):
        # Perform clustering of keywords
        groups = cluster_keywords(
            keywords,
            self.sentence_model,
            cluster_accuracy=cluster_acc,
            min_cluster_size=min_size,
        )
        # Generate new code names
        code_prompt = json_codes()
        prompt_codes = prompt_for_codes(groups, code_prompt)

        models = self.wrapper.models
        ollama_client = models.get("llm_client")
        ollama_model_name = self.wrapper.config.ollama_model
        llm_model = models.get("llm_model")
        llm_tokenizer = models.get("llm_tokenizer")
        device = self.wrapper.config.device
        dtype = self.wrapper.config.dtype

        # Define codes using LLM
        codes = generate_codes(
            prompt=prompt_codes,
            use_ollama=self.use_ollama,
            ollama_client=ollama_client,
            ollama_model_name=ollama_model_name,
            wrapper=self.wrapper,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            device=device,
            dtype=dtype,
        )
        # Rename clusters
        renamed_clusters = rename_clusters(groups, codes)
        sim = calculate_default_code_similarity(
            renamed_clusters,
            self.sentence_model,
        )
        assigned = assign_codes_to_data(data, sim)

        return assigned, sim


class DeductiveCoding:
    def __init__(self, sentence_model):
        self.sentence_model = sentence_model

    def code(self, data, keywords, code_list, threshold, top_k, max_kw):
        combined_result, clusters = cluster_custom_codes(
            data,
            code_list,
            keywords,
            self.sentence_model,
            threshold,
            top_k,
            max_kw,
        )

        return combined_result, clusters
