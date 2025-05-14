from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from thematico.services.pipeline import ModelWrapper
import json
import torch
import socket, errno
from ollama import ResponseError as OlamaCompletionError
import logging

logger = logging.getLogger("uvicorn.info")


class StopSequenceCriteria(StoppingCriteria):
    """Stopping criteria based on a list of stop sequences in the generated text."""

    def __init__(self, stop_strings: list[str], tokenizer: AutoTokenizer):
        """Initialize the stopping criteria with a list of stop sequences.

        Args:
            stop_strings (List[str]): A list of stop sequences to check in the generated text.
            tokenizer (AutoTokenizer): The tokenizer used for decoding the generated text.
        """
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any
    ) -> bool:
        """Check if any of the stop sequences are present in the generated text.

        Args:
            input_ids (torch.Tensor): The input tensor containing the generated text.
            scores (torch.Tensor): The scores associated with the generated text.

        Returns:
            bool: True if any of the stop sequences are present in the generated text, False otherwise.
        """
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return any(stop_string in decoded_text for stop_string in self.stop_strings)


def json_codes() -> tuple[str, str, str]:
    """Returns a prompt with the expected output format for the codes generation task.

    Args: None
    Returns:
        tuple: A tuple containing the expected output format for the codes generation task.
    """
    prompt = """
    Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python.
    There may be more fewer items than in this example.
    It should be in the schema:

    {
        "codes": [
            {
                "<cluster_key>": "<new_name_for_that_cluster>"
            },
            {
                "no_cluster": "Miscellaneous_Topics"
            },
            ...
        ]
    }
    """
    return prompt


def prompt_for_codes(json_groups, json_prompt: dict) -> str:
    """Generates a structured prompt for an AI model to rename clusters of keywords with
    meaningful names.

    Args:
        json_groups (dict): A dictionary where keys are cluster names (e.g., "Code_1") and
        values are lists of associated keywords.

    Returns:
        str: A structured prompt for the AI model to rename the clusters of keywords
    """

    prompt = f"""
    Below is a dictionary containing clustered keywords from a study.
    Each cluster represents a theme, but it currently has generic names (e.g., Code_1, Code_2).
    One cluster is left unassigned and contains keywords that do not fit into any specific theme.
    Your task is to provide a meaningful name for each cluster (change the name of the one named "no_cluster" to "Miscellaneous_Topics") based on the keywords in it.
    For **each** key in that dictionary, provide **one** new name.
    The dictionary is structured as follows:
    {json_groups}

    Provide ONLY ONE new name for each code based on the keywords in the cluster.
    Answer me only with the new names for each code.

    Do not include the prompt in your response.
    Provide the output in JSON format.

    {json_prompt}
    """

    return prompt


def define_codes(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    role = """
    You are an expert in qualitative research and text analysis with deep expertise in generating themes for groups of text data.
    You respond to the user's request by providing new code names for each code group.
    """

    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model.to(device=device, dtype=dtype)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)

    input_id_tensors = model_inputs["input_ids"]

    generated_chunks = [
        out_ids[len(inp_id) :]
        for inp_id, out_ids in zip(input_id_tensors, generated_ids)
    ]

    final_json = tokenizer.batch_decode(generated_chunks, skip_special_tokens=True)[0]
    try:
        start = final_json.find("{")
        end = final_json.rfind("}")
        if start != -1 and end != -1:
            json_only = final_json[start : end + 1]
        else:
            json_only = final_json
        response = json.loads(json_only)

        return response
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")

        return {"error": "Invalid JSON response", "raw_output": final_json}


def json_themes() -> tuple[str, str, str]:
    """Returns a prompt wih the expected output format for the themes generation task.

    Args: None
    Returns:
        tuple: A tuple containing the expected output format for the themes generation task.
    """
    prompt = """
    Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python.
    There could be more or less than the number of themes in the example.
    It should be in the schema:

    {
        "themes": [
            {
                "theme_1": Theme_name
            },
            ...
        ]
    }
    """
    return prompt


def prompt_for_themes(keywords: list, json_prompt: dict) -> str:
    """Generates a structured prompt for an AI model to rename clusters of keywords with
    meaningful names.

    Args:
        keywords (list): A list of keywords extracted from the text data.
    Returns:
        str: A structured prompt for the AI model to rename the clusters of keywords
    """

    prompt = f"""
    Below is a list containing extracted keywords from a study.
    Your task is to identify the core themes of the study according to the keywords.
    Describe the themes in one to four words.
    The list is structured as follows:
    {keywords}

    Do not include the prompt in your response.
    Provide the output in JSON format.

    {json_prompt}
    """

    return prompt


def define_themes(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    role = """
    You are an expert in qualitative research and text analysis with deep expertise in generating themes for groups of text data.
    You respond to the user's request by providing a list of themes for the defined keywords.
    """

    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model.to(device=device, dtype=dtype)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)

    input_id_tensors = model_inputs["input_ids"]

    generated_chunks = [
        out_ids[len(inp_id) :]
        for inp_id, out_ids in zip(input_id_tensors, generated_ids)
    ]

    final_json = tokenizer.batch_decode(generated_chunks, skip_special_tokens=True)[0]
    try:
        start = final_json.find("{")
        end = final_json.rfind("}")
        if start != -1 and end != -1:
            json_only = final_json[start : end + 1]
        else:
            json_only = final_json
        response = json.loads(json_only)

        return response
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")

        return {"error": "Invalid JSON response", "raw_output": final_json}


def assign_codes_to_data(data: dict, similarity_results: dict) -> dict:
    """Assigns the generated codes to the data based on the keywords present in each
    entry.

    Args:
        data (dict): A dictionary containing a list of data entries under the key "result".
        similarity_results (dict): A dictionary where keys are code names and values are dictionaries
                                   with keywords and their similarity scores.

    Returns:
        data (dict): The updated data dictionary with the codes assigned to each entry
        based on the keywords present.
    """
    keyword_to_code = {}

    for code, details in similarity_results.items():
        keywords = details.get("code_keywords", [])
        scores = details.get("similarity_scores", [])
        for keyword, score in zip(keywords, scores):
            keyword_to_code[keyword] = (code, score)

    for entry in data["result"]:
        entry_codes = {}
        for keyword in entry["keywords"]:
            if keyword in keyword_to_code:
                code, score = keyword_to_code[keyword]
                if code not in entry_codes:
                    entry_codes[code] = {
                        "keywords": [],
                        "similarity_scores": [],
                    }
                entry_codes[code]["keywords"].append(keyword)
                entry_codes[code]["similarity_scores"].append(score)

        entry["codes"] = entry_codes

    return data


def generate_codes(
    prompt: str,
    use_ollama: bool,
    ollama_client: Any,
    ollama_model_name: str,
    wrapper: "ModelWrapper",
    llm_model,
    llm_tokenizer,
    device: str,
    dtype: str,
):
    if use_ollama:
        raw_text = None

        if ollama_client is None or ollama_model_name is None:
            raise ValueError("Ollama client or model name not provided.")

        try:
            resp = ollama_client.chat(
                model=ollama_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            raw_text = resp["message"]["content"]
            response = json.loads(raw_text)
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response", "raw_output": raw_text}

        except Exception as e:
            win_err = getattr(e, "winerror", None)
            posix_err = getattr(e, "errno", None)

            if win_err == 10060 or posix_err in (errno.ETIMEDOUT, errno.EAGAIN):
                logger.warning(
                    "Ollama connection timed out; falling back to local LLM."
                )
            else:
                logger.warning(
                    f"Ollama unreachable ({e!r}); falling back to local LLM."
                )

    if llm_model is None or llm_tokenizer is None:
        wrapper._load_llm_locally()
        llm_model = wrapper.models["llm_model"]
        llm_tokenizer = wrapper.models["llm_tokenizer"]
    try:
        codes = define_codes(
            llm_model,
            llm_tokenizer,
            prompt,
            device=device,
            dtype=dtype,
        )

        return codes
    except Exception as e:
        logger.error(f"Error generating codes with LLM: {e}")
        return {}


def generate_themes(
    prompt: str,
    use_ollama: bool,
    ollama_client: Any,
    ollama_model_name: str,
    llm_model,
    llm_tokenizer,
    wrapper: "ModelWrapper",
    device: str,
    dtype: str,
):
    raw_text = ""

    if use_ollama:
        if ollama_client is None or ollama_model_name is None:
            raise ValueError("Ollama client or model name not provided.")

        try:
            resp = ollama_client.chat(
                model=ollama_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            raw_text = resp["message"]["content"]

            response = json.loads(raw_text)
            return response

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {e}")
            return {"error": "Invalid JSON response", "raw_output": raw_text}

        except Exception as e:
            win_err = getattr(e, "winerror", None)
            posix_err = getattr(e, "errno", None)

            if win_err == 10060 or posix_err in (errno.ETIMEDOUT, errno.EAGAIN):
                logger.warning(
                    "Ollama connection timed out; falling back to local LLM."
                )
            else:
                logger.warning(
                    f"Ollama unreachable ({e!r}); falling back to local LLM."
                )

        if llm_model is None or llm_tokenizer is None:
            wrapper._load_llm_locally()
            llm_model = wrapper.models["llm_model"]
            llm_tokenizer = wrapper.models["llm_tokenizer"]
        try:
            themes = define_themes(
                llm_model,
                llm_tokenizer,
                prompt,
                device=device,
                dtype=dtype,
            )

            return themes
        except Exception as e:
            logger.error(f"Error generating themes with LLM: {e}")
