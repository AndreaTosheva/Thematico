import logging
import pandas as pd
import requests
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import re
import io
from PIL import Image
import streamlit as st
import urllib.parse
from typing import Union, IO
from pathlib import Path

logger = logging.getLogger("uvicorn.info")


def csv_to_json(
    input_csv: Union[str, Path],
    text_col: str = "text",
    speaker_col: str = "speaker",
    start_col: str = "start",
    end_col: str = "end",
) -> None:
    """Convert a CSV file to a JSON file with a specific format.

    Args:
        input_csv (Union[str, Path]): Path to the input CSV file.
        output_json (Union[str, Path]): Path to the output JSON file.
        text_col (str): Name of the column containing text data. Defaults to "text".
        speaker_col (str): Name of the column containing speaker data. Defaults to "speaker".
        start_col (str): Name of the column containing start time data. Defaults to "start".
        end_col (str): Name of the column containing end time data. Defaults to "end".
    """
    df = pd.read_csv(input_csv, dtype=str)

    cols = []
    for c in [text_col, start_col, end_col, speaker_col]:
        if c in df.columns:
            cols.append(c)

    df = df[cols]
    df = df[df[text_col].notna() & df[text_col].str.strip().ne("")]

    results = []
    for _, row in df.iterrows():
        entry = {"text": row[text_col].strip()}
        if (
            start_col in df.columns
            and pd.notna(row[start_col])
            and row[start_col].strip()
        ):
            try:
                entry["start"] = float(row[start_col])
            except ValueError:
                logger.warning(f"Invalid start time '{row[start_col]}' for row {row}")
                pass
        if end_col in df.columns and pd.notna(row[end_col]) and row[end_col].strip():
            try:
                entry["end"] = float(row[end_col])
            except ValueError:
                logger.warning(f"Invalid end time '{row[end_col]}' for row {row}")
                pass
        if (
            speaker_col in df.columns
            and pd.notna(row[speaker_col])
            and row[speaker_col].strip()
        ):
            entry["speaker"] = [row[speaker_col].strip()]
        results.append(entry)

    return {"result": results}


def parse_timestamps(ts: str) -> float:
    """Parse a timestamp string and convert it to seconds.

    Args:
        ts (str): Timestamp string in the format "HH:MM:SS" or "MM:SS" or "SS".

    Returns:
        float: Timestamp in seconds.
    """
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        h = 0
        m = 0
        s = parts[0]
    return h * 3600 + m * 60 + s


def txt_to_json(
    input_txt: Union[str, Path, IO],
) -> dict:
    """Convert a txt file to a JSON file with a specific format.

    Args:
        input_txt (Union[str, Path, IO]): Path to the input txt file.

    Returns:
        dict: JSON object containing the parsed data.
    """
    # Regular expressions to match the timestamp and speaker format
    ts_re = re.compile(r"^\[\s*(?P<start>\d+\.\d+)\s*[–-]\s*(?P<end>\d+\.\d+)\s*\]")
    sp_re = re.compile(r"^(?P<sp>SPEAKER_\d+):\s*")

    close_after = False
    if isinstance(input_txt, (str, Path)):
        f = open(input_txt, "r", encoding="utf-8")
        close_after = True
    else:
        f = io.TextIOWrapper(input_txt, encoding="utf-8")
        close_after = True

    results = []
    for raw_text in f:
        line = raw_text.strip()
        if not line:
            continue

        m_ts = ts_re.match(line)
        if not m_ts:
            continue
        start = float(m_ts.group("start"))
        end = float(m_ts.group("end"))

        rest = line[m_ts.end() :].lstrip()

        m_sp = sp_re.match(rest)
        if m_sp:
            speaker = m_sp.group("sp")
            text = rest[m_sp.end() :].strip()

        else:
            speaker = "Unknown speaker"
            text = rest

        results.append(
            {
                "start": start,
                "end": end,
                "speaker": [speaker],
                "text": text,
            }
        )

    if close_after:
        f.close()

    return {"result": results}


def fetch_url(raw_url: str):
    """Try to fetch JSON from raw_url; if it fails, try again with "?download=1" appended.

    Args:
        raw_url (str): URL to fetch JSON from.
    """

    tried = []
    url = raw_url.strip()

    if not urllib.parse.urlparse(url).scheme:
        url = "https://" + url

    tried.append(url)
    fallback = url + "?download=1"
    if "download=1" not in url:
        tried.append(fallback)
    else:
        fallback = None

    try:
        return read_json_file(url)
    except Exception as e1:
        if fallback:
            try:
                return read_json_file(fallback)
            except Exception as e2:
                raise ValueError(
                    f"Could not fetch JSON from any of the URLs: {tried}"
                ) from e2
        raise ValueError(f"Could not fetch JSON from any of the URLs: {tried}") from e1


def read_json_file(url: str) -> dict:
    """Fetch a JSON file from a given URL, handling redirects and content validation.

    Args:
        url (str): Public URL of the JSON file

    Returns:
        response.json() (dict): JSON content of the file
    """
    try:
        session = requests.Session()
        response = session.get(url, allow_redirects=True)
        response.raise_for_status()

        if not response.text.strip():
            logger.error("Empty response content.")
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type:
            logger.error(f"Invalid content type: {content_type}")
            return None
        else:
            logger.info(f"Content type: {content_type}")
            return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"Error reading JSON file: {e}")
        return None
    except ValueError as e:
        logger.error(f"JSON decode error: {e}")
        return None


def validate_json_format(json_data):
    """
    Validates if the given JSON data follows the expected format.

    Parameters:
    json_data (dict): JSON data to validate.

    Returns:
    bool: True if valid, False otherwise.
    """
    if not isinstance(json_data, dict):
        return False

    if "result" in json_data:
        data = json_data["result"]

    elif "transcription" in json_data:
        data = json_data["transcription"]

    else:
        return False

    if not isinstance(data, list):
        return False

    for entry in data:
        if not isinstance(entry, dict):
            return False

        if "speaker" not in entry or "text" not in entry:
            return False

        if not isinstance(entry["text"], str):
            return False

    return True


def generate_wordcloud(
    keywords: list,
    mask_path: str,
    background_color: str = "#F3E9DC",
    width: int = 800,
    height: int = 400,
    colormap: str = "viridis",
    max_words: int = 200,
    contour_color: str = "black",
    contour_width: int = 0.1,
):
    """Generates a word cloud from the keywords extracted from the transcription.

    Args:
        background_color (str, optional): The background color of the wordcloud. Defaults to "white".
        width (int, optional): Width of the wordcloud. Defaults to 800.
        height (int, optional): Height of the wordcloud. Defaults to 400.
        colormap (str, optional): Colormap of the wordcloud. Defaults to "viridis".
        max_words (int, optional): Maximum words to be displayed. Defaults to 200.
        contour_color (str, optional): Color of the contour. Defaults to "steelblue".
        contour_width (int, optional): Width of the contour. Defaults to 1.
    """

    text = " ".join(keywords)

    mask = np.array(Image.open(mask_path).convert("L"))
    binary_mask = np.where(mask > 128, 255, 0).astype(np.uint8)

    wc = WordCloud(
        background_color=background_color,
        width=width,
        height=height,
        colormap=colormap,
        max_words=max_words,
        contour_color=contour_color,
        contour_width=contour_width,
        mask=binary_mask,
    )
    wc.generate(text)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    with st.container():
        st.pyplot(fig, use_container_width=False)


def custom_pipeline_output(mode: str = "inductive") -> dict:
    """

    Returns:
        _type_: _description_
    """
    entries = st.session_state.data["result"]
    if mode == "inductive":
        df = st.session_state.get("inductive_df", pd.DataFrame())

    elif mode == "deductive":
        df = st.session_state.get("deductive_df", pd.DataFrame())

    else:
        logger.error("Invalid mode. Use 'inductive' or 'deductive'.")
        return {"result": []}
    if df.empty:
        logger.warning("No keywords found in the DataFrame.")
        return {"result": []}

    code_map = {
        row["Code"]: [kw.strip() for kw in row["Keywords"].split(",") if kw.strip()]
        for _, row in df.iterrows()
    }

    result = []
    for entry in entries:
        keywords = entry.get("keywords", [])
        codes_entry = {
            code: [kw for kw in keywords if kw in kws]
            for code, kws in code_map.items()
            if any(kw in keywords for kw in kws)
        }
        result.append(
            {
                "start": entry.get("start", None),
                "end": entry.get("end", None),
                "speaker": entry.get("speaker", []),
                "text": entry["text"],
                "keywords": keywords,
                "codes": {
                    code: {"keywords": matched_keywords}
                    for code, matched_keywords in codes_entry.items()
                },
                "sentiment": entry.get("sentiment_score"),
            }
        )

    return {"result": result}
