from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re
from typing import Any


def cluster_keywords(
    keywords_list: list,
    model: SentenceTransformer,
    cluster_accuracy: int,
    min_cluster_size: int,
) -> dict:
    """Clusters the extracted keywords from the text. The keywords are clustered based
    on their semantic similarity.

    Args:
        keywords_list (List[str]): A list of extracted keywords.
        model (SentenceTransformer): The SentenceTransformer model used for clustering.
        cluster_accuracy (int): The threshold for clustering the keywords.
        min_cluster_size (int): The minimum size of a cluster.

    Returns:
        Dict: A dictionary containing the clusters of keywords with a temporary name.
    """
    cluster_name_list = []
    corpus_sentences_list = []

    corpus_set = set(keywords_list)
    cluster_accuracy = cluster_accuracy / 100

    df_final = pd.DataFrame(columns=["Cluster_Name", "Keyword"])
    while True:
        corpus_sentences = list(corpus_set)
        check_len = len(corpus_sentences)

        corpus_embeddings = model.encode(
            corpus_sentences, convert_to_tensor=True, show_progress_bar=True
        )
        clusters = util.community_detection(
            corpus_embeddings,
            threshold=cluster_accuracy,
            min_community_size=min_cluster_size,
        )

        for keyword, cluster in enumerate(clusters):
            temp_cluster_name = f"Temp_cluster_{keyword + 1}"
            for sentence_id in cluster:
                corpus_sentences_list.append(corpus_sentences[sentence_id])
                cluster_name_list.append(temp_cluster_name)

        df_new = pd.DataFrame(
            {"Cluster_Name": cluster_name_list, "Keyword": corpus_sentences_list}
        )

        df_final = pd.concat([df_final, df_new], ignore_index=True)

        have = set(df_new["Keyword"])
        corpus_set = corpus_set - have
        remaining = len(corpus_set)

        if check_len == remaining:
            break

        unique_clusters = sorted(
            df_final["Cluster_Name"].dropna().unique(),
            key=lambda x: int(re.search(r"(\d+)$", x).group(1)),
        )
        cluster_mapping = {
            old_name: f"Code_{i+1}" for i, old_name in enumerate(unique_clusters)
        }

        df_final["Cluster_Name"] = df_final["Cluster_Name"].map(cluster_mapping)

        def sort_key(cluster: str) -> int:
            """Extracts the numerical part of a cluster name (e.g., "Code_1", "Code_2")
            and returns it as an integer for sorting purposes.

            Args:
                cluster (str): The cluster name in the format "Code_X",
                where X is a number.

            Returns:
                int: The extracted numerical value if found,
                otherwise returns float('inf') to ensure unrecognized clusters are sorted last.
            """
            match = re.search(r"Code_(\d+)", cluster)
            return int(match.group(1)) if match else float("inf")

        json_groups = {
            k: v
            for k, v in sorted(
                df_final.groupby("Cluster_Name")["Keyword"].apply(list).items(),
                key=lambda x: sort_key(x[0]),
            )
        }

        json_groups["no_cluster"] = list(corpus_set) if corpus_set else []

        return json_groups


def rename_clusters(json_groups: dict[str, list[str]], new_names_json: dict) -> dict:
    """Rename the clusters of keywords with the cleaned codes.

    Args:
        json_groups (dict): A dictionary where keys are current cluster names (e.g., "Code_1")
        and values are lists of associated keywords.
        cleaned_codes (list[str]): A list of new cluster names in the same order as the original clusters.

    Returns:
        json_groups_renamed (dict[str, list[str]]): A dictionary where clusters are renamed
        based on cleaned_codes, while unclustered items (if present) retain their original key.
    """
    new_name_mapping = {
        list(entry.keys())[0]: list(entry.values())[0]
        for entry in new_names_json.get("codes", [])
    }
    json_groups_renamed = {
        new_name_mapping.get(k, k): v for k, v in json_groups.items()
    }
    return json_groups_renamed


def cluster_custom_codes(
    combined_result: dict[str, Any],
    code_list: list,
    keywords: list,
    sentence_model: SentenceTransformer,
    threshold: float = 0.4,
    top_k: int = 25,
    max_keywords: int = 10,
) -> dict:
    """Group the keywords based on their similarity to given user-passed codes and append the clusters back to the result dictionary.

    Args:
        combined_result (dict): The combined result of keyword extraction and code definition.
        code_list (list): A list of user-passed codes.
        keywords (list): A list of keywords from the transcription.
        sentence_model (SentenceTransformer): A pre-trained sentence transformer model for encoding.
        threshold (float, optional): Minimum similarity score to consider a keyword relevant.. Defaults to 0.4.
        top_k (int, optional): Number of top similar keywords to consider for each code. Defaults to 25.
        max_keywords (int, optional): Maximum number of keywords to assign to each code. Defaults to 10.

    Returns:
        dict: A dictionary where:
            - The key is the code.
            - The value is a list of tuples where each tuple contains the keyword and its similarity score
    """
    if not code_list or not keywords:
        return {}

    clusters = {}
    keyword_embeddings = sentence_model.encode(keywords, convert_to_tensor=True)

    for code in code_list:
        try:
            code_embeddings = sentence_model.encode(code, convert_to_tensor=True)
            similarities = util.cos_sim(code_embeddings, keyword_embeddings)[0]
            top_indices = similarities.argsort(descending=True)[
                : min(top_k, len(keywords))
            ]

            filtered_keywords = []
            filtered_scores = []

            for i in top_indices:
                score = round(similarities[i].item(), 2)
                if score >= threshold:
                    keyword = keywords[i]
                    filtered_keywords.append(keyword)
                    filtered_scores.append(score)

            # Stores the keywords and their similarity scores for the current code
            clusters[code] = {
                "code_keywords": filtered_keywords,
                "similarity_scores": filtered_scores,
            }
        except Exception as e:
            clusters[code] = {
                "code_keywords": [],
                "similarity_scores": [],
            }
    # Iterate through the combined result and access the "result" key
    for item in combined_result:
        # For each code iterate through keywords and scores
        for code, data in clusters.items():
            matched_keywords = []
            matched_scores = []
            # Get the keywords from the item
            item_keywords = set(item.get("keywords", []))
            # Iterate over the paired keywords and similarity scores from the cluster data
            for keyword, score in zip(data["code_keywords"], data["similarity_scores"]):
                if keyword in item_keywords:
                    matched_keywords.append(keyword)
                    matched_scores.append(score)

            if matched_keywords:
                item["codes"][code] = {
                    "code_keywords": matched_keywords,
                    "similarity_scores": matched_scores,
                }

    return combined_result, clusters


def calculate_default_code_similarity(
    renamed_clusters: dict, model: SentenceTransformer
) -> dict:
    """Calculates the similarity scores between the renamed clusters and their keywords.

    Args:
        renamed_clusters (dict): A dictionary where keys are cluster names (str) and values are
                                lists of associated keywords (List[str])
        model (SentenceTransformer): A pre-trained SentenceTransformer model used to encode
                                     the text into embeddings.

    Returns:
        dict: A dictionary containing similarity results for each cluster. Each entry includes:
            - "code_keywords": The list of keywords for the cluster.
            - "similarity_scores": A list of cosine similarity scores (rounded to 2 decimals)
                                   between the cluster name and each keyword.
            If an error occurs during processing, the keyword list and scores will be empty.
    """

    similarity_results = {}

    for cluster_name, keywords in renamed_clusters.items():
        try:
            code_embeddings = model.encode(cluster_name, convert_to_tensor=True)
            keyword_embeddings = model.encode(keywords, convert_to_tensor=True)

            similarities = util.cos_sim(code_embeddings, keyword_embeddings)[0]

            scores = [round(similarity.item(), 2) for similarity in similarities]

            similarity_results[cluster_name] = {
                "code_keywords": keywords,
                "similarity_scores": scores,
            }
        except Exception as e:
            similarity_results[cluster_name] = {
                "code_keywords": [],
                "similarity_scores": [],
            }

    return similarity_results
