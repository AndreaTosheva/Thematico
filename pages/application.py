import os
import io
import re
import sys
import json
import logging

import streamlit as st
from collections import defaultdict
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from wordcloud import WordCloud

import ollama
from keybert import KeyBERT
import pages.utils.nav as n
import pages.utils.styles as styles

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./thematico"))
sys.path.append(os.path.abspath("./pages/utils"))

from thematico.keywords import (
    highlight_keywords,
)

from thematico.io import (
    read_json_file,
    csv_to_json,
    validate_json_format,
    txt_to_json,
    fetch_url,
    generate_wordcloud,
    custom_pipeline_output,
)
from thematico.clustering import calculate_default_code_similarity


from thematico.services.pipeline import (
    PipelineConfig,
    ModelWrapper,
    run_pipeline,
    analyze_custom_string,
)
from thematico.services.strategies import InductiveCoding, DeductiveCoding

st.set_page_config(
    page_title="Thematico",
    page_icon=":bar_chart:",
    layout="wide",
)

logger = logging.getLogger("uvicorn.info")

ollama_url = st.secrets["ollama"]["url"]
ollama_model = st.secrets["ollama"]["model"]
hf_token = st.secrets["huggingface"]["token"]
default_mode = st.secrets["pipeline"]["default_mode"]

if "cfg" not in st.session_state:
    st.session_state.cfg = PipelineConfig(
        mode=default_mode,
        use_ollama=True,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        hf_token=hf_token,
    )
wrapper = ModelWrapper(st.session_state.cfg)
wrapper.load_models(st.session_state.cfg.use_ollama)
models = wrapper.get_models(use_ollama=st.session_state.cfg.use_ollama)
st.session_state.models = models


# ------------------ Page setup ------------------


st.markdown("""<div class="main-container-app">""", unsafe_allow_html=True)
styles.apply_styles()
n.run(index=2)

# ------------------ Load and validate files -----------------

st.markdown(
    """<div class="app-container">
    <p class='subtitle'>Step 1: Load the transcribed audio file</p>
    </div>""",
    unsafe_allow_html=True,
)
with st.expander("💡File Format Info", expanded=False):
    st.markdown(
        """<p class='body-app-text'>The transcribed audio file could be passed as an input in 3 formats: JSON, CSV and TXT. In case you are working with JSON format it can be a local file or a public URL.</br>
        Below you can find examples of the required format for each of the file types:</br>
        </p>""",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            "<div style='background:#264653; color:white; padding:8px; border-radius:5px; "
            "text-align:center; margin-bottom:0.5rem;'>JSON</div>",
            unsafe_allow_html=True,
        )
        example_json = {
            "result": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "speaker": "Speaker 1",
                    "text": "Hello, how are you?",
                },
                {
                    "start": 3.0,
                    "end": 5.0,
                    "speaker": "Speaker 2",
                    "text": "I am fine, thank you!",
                },
            ]
        }
        st.code(json.dumps(example_json, indent=2), language="json")
    with col2:
        st.markdown(
            "<div style='background:#6A994E; color:white; padding:8px; border-radius:5px; "
            "text-align:center; margin-bottom:0.5rem;'>CSV</div>",
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(
            {
                "start (Optional)": [0.0, 3.0, 5.5, 8.0],
                "end (Optional)": [2.0, 5.0, 7.0, 10.0],
                "speaker": ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"],
                "text": [
                    "Hello, how are you?",
                    "I am fine, thank you!",
                    "What brings you here?",
                    "Just wanted to chat.",
                ],
            }
        )
        styled = df.style.set_properties(
            **{
                "background-color": "#F2E8CF",
                "color": "#333333",
                "border": "1px solid #ccc",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("background-color", "#52796F"), ("color", "white")],
                }
            ]
        )
        st.dataframe(styled, use_container_width=True)
    with col3:
        st.markdown(
            "<div style='background:#BC4749; color:white; padding:8px; border-radius:5px; "
            "text-align:center; margin-bottom:0.5rem;'>TXT</div>",
            unsafe_allow_html=True,
        )
        txt_example = """
        [00:00.000 – 00:02.000] Speaker 1: Hello, how are you?
        [00:03.000 – 00:05.000] Speaker 2: I am fine, thank you!
        [00:05.500 – 00:08.000] Speaker 1: What brings you here?
        [00:08.000 – 00:10.000] Speaker 2: Just wanted to chat.
        [00:10.000 – 00:12.000] Speaker 1: That's great!
        """
        st.code(txt_example, language="text")

uploaded_file = st.file_uploader(
    "Upload your transcribed audio file",
    type=["txt", "csv", "json"],
    label_visibility="collapsed",
)
st.markdown(
    "<p class='body-app-text'>Or enter a public JSON file URL</p>",
    unsafe_allow_html=True,
)

url = st.text_input("url", label_visibility="collapsed")

DEFAULTS = {
    "edit_mode": False,
    "keyword_extraction": False,
    "analysis_mode": False,
    "custom_analysis_mode": False,
    "show_themes": False,
    "sentiment_mode": False,
    "inductive": False,
    "deductive": False,
    "wordcloud": False,
    "edit_kw_mode": False,
    "rerun_pipeline": False,
    "data": None,
    "updated_data": None,
    "keywords": None,
    "models": None,
    "themes": None,
    "inductive_pipeline_output": None,
    "deductive_pipeline_output": None,
    "similarity_scores": None,
    "clusters": None,
    "custom_analysis_result": None,
    "inductive_df": pd.DataFrame(),
    "deductive_df": pd.DataFrame(),
    "theme_df": pd.DataFrame(),
}
for key, default in DEFAULTS.items():
    st.session_state.setdefault(key, default)

if not uploaded_file and not url:
    for k in [
        "data",
        "updated_data",
        "keywords",
        "models",
        "themes",
        "inductive_pipeline_output",
        "deductive_pipeline_output",
        "similarity_scores",
        "clusters",
        "custom_analysis_result",
    ]:
        st.session_state[k] = None

    for flag in [
        "edit_mode",
        "keyword_extraction",
        "analysis_mode",
        "custom_analysis_mode",
        "show_themes",
        "sentiment_mode",
        "inductive",
        "deductive",
        "wordcloud",
        "edit_kw_mode",
        "rerun_pipeline",
    ]:
        st.session_state.setdefault(flag, False)


if st.session_state.data is None:
    if uploaded_file:
        if uploaded_file.name.lower().endswith(".csv"):
            loaded = csv_to_json(uploaded_file)

        elif uploaded_file.name.lower().endswith(".txt"):
            loaded = txt_to_json(uploaded_file)

        elif uploaded_file.name.lower().endswith(".json"):
            loaded = json.load(uploaded_file)

    elif url:
        try:
            loaded = fetch_url(url)
        except Exception as e:
            logger.error(f"Error fetching URL: {e}")
            st.stop()
    else:
        st.stop()

    if not validate_json_format(loaded):
        st.error("Invalid JSON format. Please check the file.")
        st.stop()

    st.session_state.data = loaded


data = st.session_state.data

if st.session_state.analysis_mode and st.session_state.get("rerun_pipeline", False):
    data_to_analyze = st.session_state.updated_data or st.session_state.data
    (
        st.session_state.inductive_pipeline_output,
        st.session_state.keywords,
        st.session_state.themes,
        st.session_state.similarity_scores,
        st.session_state.clusters,
    ) = run_pipeline(
        data_to_analyze,
        st.session_state.models,
        st.session_state.cfg,
        wrapper,
        skip_keyword_extraction=True,
    )
    st.session_state.rerun_pipeline = False

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = False

if not st.session_state.analysis_mode and not st.session_state.edit_mode:
    st.markdown(
        """<div class="app-container">
    <p class='subtitle'>Step 2: Transcription editing</p>
    </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("💡Transcription Info", expanded=False):
        st.markdown(
            """<p class='body-app-text'>The transcription is displayed below. Read carefully and edit it if needed.</br>
            When you are ready to proceed to analysis, click the "Analyze" button and come back to the beginning of the text.</br>
            </p>""",
            unsafe_allow_html=True,
        )

    st.markdown("<p class='steps'>Transcription</p>", unsafe_allow_html=True)
    for i, entry in enumerate(data.get("result", [])):
        raw_speaker = entry.get("speaker", ["Unknown speaker"])
        if isinstance(raw_speaker, list):
            speaker = raw_speaker[0]
        else:
            speaker = raw_speaker
        start_time = entry.get("start", 0.0)
        end_time = entry.get("end", 0.0)
        text = entry.get("text", "No text available")

        st.markdown(
            f"""
            <div class="custom-container-speaker">
                <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Edit transcription"):
            st.session_state.edit_mode = True
            st.rerun()
    with col2:
        if st.button("Analyze transcription"):
            data_to_analyze = st.session_state.updated_data or st.session_state.data
            (
                st.session_state.inductive_pipeline_output,
                st.session_state.keywords,
                st.session_state.themes,
                st.session_state.similarity_scores,
                st.session_state.clusters,
            ) = run_pipeline(
                data_to_analyze,
                st.session_state.models,
                st.session_state.cfg,
                wrapper,
                skip_keyword_extraction=False,
            )
            st.session_state.analysis_mode = True
            st.rerun()

    st.stop()


# —————————————————————————————————————————————
# 3. Edit transcription
# —————————————————————————————————————————————


if (
    not st.session_state.keyword_extraction
    and not st.session_state.edit_mode
    and not st.session_state.analysis_mode
    and not st.session_state.sentiment_mode
):

    st.markdown(
        "<p class='steps'>Transcription</p>",
        unsafe_allow_html=True,
    )
    for i, entry in enumerate(data.get("result", [])):
        speaker = entry.get("speaker", ["Unknown speaker"])[0]
        start_time = entry.get("start", 0.0)
        end_time = entry.get("end", 0.0)
        text = entry.get("text", "No text available")

        st.markdown(
            f"""
            <div class="custom-container-speaker">
                <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )


# render or edit
elif st.session_state.edit_mode and not st.session_state.analysis_mode:
    col1, col2 = st.columns(2)
    edited_entries = []

    with col1:
        st.markdown("<p class='steps'>Transcription</p>", unsafe_allow_html=True)
        for i, entry in enumerate(data.get("result", [])):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")

            st.markdown(
                f"""
                <div class="custom-container-speaker">
                    <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            "<p class='steps'>Edit Transcription</p>",
            unsafe_allow_html=True,
        )

        for i, entry in enumerate(data.get("result", [])):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")

            st.markdown(
                f"""
                <div class="custom-container-speaker">
                    <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s):
                </div>
                """,
                unsafe_allow_html=True,
            )
            num_lines = text.count("\n") + text.count(". ") + 5
            height = min(max(100, num_lines * 18), 600)

            st.markdown(f'<div class="custom-textbox">', unsafe_allow_html=True)
            edited_text = st.text_area(
                label=f"Edit line {i+1}",
                value=text,
                key=f"edit_{i}",
                label_visibility="collapsed",
                height=height,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            edited_entries.append(
                {
                    "speaker": [speaker],
                    "start": start_time,
                    "end": end_time,
                    "text": edited_text,
                }
            )

        edited_json = json.dumps(
            {"result": edited_entries}, ensure_ascii=False, indent=2
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                label="Download edited transcription",
                data=edited_json,
                file_name="edited_transcription.json",
                mime="application/json",
            )

        with col_b:
            if st.button("Save changes"):
                st.session_state.data["result"] = edited_entries
                st.session_state.updated_data = st.session_state.data
                st.session_state.edit_mode = False
                st.rerun()


updated_data = st.session_state.updated_data
if updated_data is None:
    updated_data = data

if st.session_state.analysis_mode:
    with st.expander("💡Analysis Info", expanded=False):
        st.markdown(
            """<p class='body-app-text'> In this step, you can explore your transcript through sentiment analysis and keyword extraction. </br>
            Sentiment analysis highlights the emotional tone of each statement, including whether it is positive, neutral, or negative, along with a confidence score and an indication of uncertainty. </br>
            Keyword extraction displays important terms from each segment of the transcript. </br>
            When only keyword extraction is active (without sentiment analysis), you can edit the keywords directly by clicking “Edit keywords,” making changes, and then saving them. </br>
            You can enable or close either feature at any time to focus on the analysis you need.
            </p>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        """<div class="app-container">
    <p class='subtitle'>Step 3: Sentiment analysis and keyword extraction</p>
    </div>""",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Click to see sentiment analysis"):
            st.session_state.sentiment_mode = True

    with col2:
        if st.button("Click to get help with keyword extraction"):
            st.session_state.keyword_extraction = True

    data_entries = st.session_state.updated_data or st.session_state.data
    if not st.session_state.keyword_extraction and not st.session_state.sentiment_mode:
        for entry in data_entries.get("result", []):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")

            st.markdown(
                f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {text}
                    </div>
                    """,
                unsafe_allow_html=True,
            )

    elif st.session_state.keyword_extraction and st.session_state.sentiment_mode:
        close_cols = st.columns(2)
        with close_cols[0]:
            if st.button("Close sentiment panel", key="close_sentiment_panel"):
                st.session_state.sentiment_mode = False
        with close_cols[1]:
            if st.button("Close keyword panel", key="close_kw_panel"):
                st.session_state.keyword_extraction = False

        # Both keyword extraction and sentiment analysis mode -> 3 columns
        for i, entry in enumerate(data.get("result", [])):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")
            keywords = entry.get("keywords", [])
            sentiment = entry.get("sentiment_score", {})
            sentiment_text = (
                f"{sentiment.get('overall_sentiment', {})} "
                f"(Confidence: {sentiment.get('confidence', 0):.2f}, "
                f"Uncertain: {sentiment.get('uncertain', False)})"
            )

            highlighted_text = highlight_keywords(text, keywords)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:

                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {sentiment_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {highlighted_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {keywords}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Keyword extraction mode
    elif st.session_state.keyword_extraction:
        data_entries = st.session_state.updated_data or st.session_state.data
        if not updated_data:
            st.error("❌ No data available for keyword extraction.")
            st.stop()
        close_cols = st.columns(2)
        if not st.session_state.get("edit_kw_mode", False):
            with close_cols[0]:
                if st.button("Edit keywords"):
                    st.session_state.edit_kw_mode = True
                    st.rerun()
        else:
            if st.button("Cancel editing", key="cancel_edit_kw"):
                st.session_state.edit_kw_mode = False
                st.rerun()

        with close_cols[1]:
            if st.button("Close keyword panel", key="close_kw_panel"):
                st.session_state.keyword_extraction = False

        for i, entry in enumerate(data.get("result", [])):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")
            keywords = entry.get("keywords", [])
            highlighted_text = highlight_keywords(text, keywords)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {highlighted_text}
                    </div>

                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                updated_data = st.session_state.updated_data
                if st.session_state.edit_kw_mode:
                    st.text_input(
                        label=f"Keywords for line {i+1} (comma-separated)",
                        value=", ".join(keywords),
                        key=f"edit_kw_{i}",
                        label_visibility="hidden",
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="custom-container-speaker">
                            <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                            {keywords}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        if st.session_state.get("edit_kw_mode", False):
            if st.button("Save keywords", key="save_keywords"):
                for i, entry in enumerate(data_entries.get("result", [])):
                    updated_keywords = st.session_state.get(f"edit_kw_{i}", [])
                    data["result"][i]["keywords"] = [
                        kw.strip() for kw in updated_keywords.split(",")
                    ]
                st.session_state.updated_data = data_entries
                st.session_state.edit_kw_mode = False
                st.session_state.keywords = updated_keywords
                st.session_state.rerun_pipeline = True
                st.rerun()

    elif st.session_state.sentiment_mode:
        if st.button("Close sentiment panel", key="close_sentiment_panel"):
            st.session_state.sentiment_mode = False

        for i, entry in enumerate(data.get("result", [])):
            speaker = entry.get("speaker", ["Unknown speaker"])[0]
            start_time = entry.get("start", 0.0)
            end_time = entry.get("end", 0.0)
            text = entry.get("text", "No text available")
            sentiment = entry.get("sentiment_score", {})
            sentiment_text = (
                f"{sentiment.get('overall_sentiment', 'N/A')} "
                f"(Confidence: {sentiment.get('confidence', 0):.2f}, "
                f"Uncertain: {sentiment.get('uncertain', False)})"
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {sentiment_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="custom-container-speaker">
                        <b>{speaker}</b> ({start_time:.1f}s - {end_time:.1f}s): <br>
                        {text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
if st.session_state.keyword_extraction:
    if st.button("Generate wordcloud", key="wordcloud_button"):
        st.session_state.wordcloud = True

if st.session_state.wordcloud:
    keywords = st.session_state.keywords
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        generate_wordcloud(keywords, mask_path="./assets/mask.png", colormap="plasma")

if st.session_state.keyword_extraction:
    st.markdown(
        """<div class="app-container">
        <p class='subtitle'>Step 4: Choose between inductive and deductive coding</p>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("💡Coding Info", expanded=False):
        st.markdown(
            """<p class='body-app-text'>In this step, you can choose between inductive or deductive coding methods to organize and analyze your data.</br>
            Inductive coding automatically suggests codes based on keyword similarities, allowing you to explore emerging themes.</br>
            Deductive coding, on the other hand, lets you define your own codes and matches them to relevant keywords in the transcript.</br>
            In both modes, the list of codes and the keywords assigned to them are fully editable. </br>
            You can adjust the coding framework by editing the table directly, ensuring the analysis reflects your specific goals or theoretical framework.
            </p>""",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inductive coding"):
            st.session_state.inductive = True
            st.session_state.deductive = False

    with col2:
        if st.button("Deductive coding"):
            st.session_state.inductive = False
            st.session_state.deductive = True
    if st.session_state.inductive:
        table_data = []
        for code, details in st.session_state.similarity_scores.items():
            kws = details.get("code_keywords", [])
            sims = details.get("similarity_scores", [])
            table_data.append(
                [
                    code,
                    ", ".join(kws),
                    ", ".join(map(str, sims)),
                ]
            )

        df_codes_inductive = pd.DataFrame(
            table_data, columns=["Code", "Keywords", "Similarity Scores"]
        )
        if (
            "inductive_edit_buffer" not in st.session_state
            or st.session_state.inductive_df.empty
        ) and not df_codes_inductive.empty:
            st.session_state.inductive_df = df_codes_inductive.copy()
            st.session_state.inductive_edit_buffer = df_codes_inductive.copy()
            st.session_state.inductive_changes_saved = False
        if not st.session_state.get("inductive_changes_saved", False):
            edited_df_inductive = st.data_editor(
                st.session_state.inductive_edit_buffer,
                key="inductive_editor",
                num_rows="dynamic",
                use_container_width=True,
            )

            if st.button("Save changes", key="save_inductive"):
                st.session_state.inductive_edit_buffer = edited_df_inductive.copy()
                updated_ind_mapping = {
                    row["Code"]: [
                        kw.strip() for kw in row["Keywords"].split(",") if kw.strip()
                    ]
                    for _, row in edited_df_inductive.iterrows()
                }

                model = st.session_state.models["sentence_model"]
                updated_scores = calculate_default_code_similarity(
                    updated_ind_mapping, model
                )
                st.session_state.similarity_scores = updated_scores
                updated_table_data = [
                    [
                        code,
                        ", ".join(details.get("code_keywords", [])),
                        ", ".join(map(str, details.get("similarity_scores", []))),
                    ]
                    for code, details in updated_scores.items()
                ]

                df_ind_updated = pd.DataFrame(
                    updated_table_data,
                    columns=["Code", "Keywords", "Similarity Scores"],
                )
                st.session_state.inductive_df = df_ind_updated.copy()
                st.session_state.inductive_edit_buffer = df_ind_updated.copy()
                st.session_state.inductive_changes_saved = True
        else:
            st.data_editor(
                st.session_state.inductive_df,
                num_rows="dynamic",
                use_container_width=True,
                disabled=True,
                key="inductive_saved_view",
            )

    if st.session_state.deductive:
        st.markdown(
            "<p class='body-app-text'>Enter the codes you want to search for (comma-separated)</p>",
            unsafe_allow_html=True,
        )
        user_codes = st.text_input(
            "Enter the codes",
            key="deductive_codes",
            label_visibility="collapsed",
            placeholder="Code1, Code2, Code3",
        )
        if st.button("Submit", key="deductive_submit"):
            if user_codes:
                code_list = [code.strip() for code in user_codes.split(",")]
                st.session_state.cfg.mode = "deductive"
                st.session_state.cfg.custom_codes = code_list

                text_keywords = st.session_state.keywords
                entries = (
                    st.session_state.updated_data["result"]
                    if st.session_state.updated_data is not None
                    else st.session_state.data["result"]
                )
                sentence_model = st.session_state.models["sentence_model"]
                deductive = DeductiveCoding(sentence_model)
                (
                    st.session_state.deductive_pipeline_output,
                    st.session_state.clusters,
                ) = deductive.code(
                    entries,
                    code_list,
                    text_keywords,
                    st.session_state.cfg.deductive_threshold,
                    st.session_state.cfg.deductive_top_k,
                    st.session_state.cfg.deductive_max_kw,
                )
                st.rerun()
            else:
                st.error("Please enter at least one code.")

        if st.session_state.clusters and len(st.session_state.clusters) > 0:
            deductive_table_data = defaultdict(
                lambda: {"keywords": [], "similarity_scores": []}
            )
            for keyword, data in st.session_state.clusters.items():
                codes = data.get("code_keywords", [])
                scores = data.get("similarity_scores", [])

                for i, code in enumerate(codes):
                    deductive_table_data[code]["keywords"].append(keyword)
                    deductive_table_data[code]["similarity_scores"].append(scores[i])

            table_data = []
            for code, values in deductive_table_data.items():
                keywords_str = ", ".join(values["keywords"])
                scores_str = ", ".join(map(str, values["similarity_scores"]))
                table_data.append([code, keywords_str, scores_str])

            df_codes_deductive = pd.DataFrame(
                table_data, columns=["Code", "Keywords", "Similarity Scores"]
            )
            if (
                "deductive_edit_buffer" not in st.session_state
                or st.session_state.deductive_df.empty
            ) and not df_codes_deductive.empty:
                st.session_state.deductive_df = df_codes_deductive.copy()
                st.session_state.deductive_edit_buffer = df_codes_deductive.copy()
                st.session_state.deductive_changes_saved = False
            if (
                "deductive_changes_saved" not in st.session_state
                or not st.session_state.deductive_changes_saved
            ):

                edited_df_deductive = st.data_editor(
                    st.session_state.deductive_edit_buffer,
                    key="deductive_editor",
                    num_rows="dynamic",
                    use_container_width=True,
                )
                if st.button("Save changes", key="save_deductive"):
                    st.session_state.deductive_edit_buffer = edited_df_deductive

                    updated_ded_mapping = {}
                    for _, row in edited_df_deductive.iterrows():
                        code = row["Code"]
                        keywords = [
                            kw.strip()
                            for kw in row["Keywords"].split(",")
                            if kw.strip()
                        ]
                        updated_ded_mapping[code] = keywords
                    model = st.session_state.models["sentence_model"]
                    updated_ded_scores = calculate_default_code_similarity(
                        updated_ded_mapping, model
                    )
                    st.session_state.similarity_scores = updated_ded_scores
                    st.session_state.deductive_df = edited_df_deductive.copy()

                    new_clusters = defaultdict(
                        lambda: {"code_keywords": [], "similarity_scores": []}
                    )
                    for code, keywords in updated_ded_mapping.items():
                        scores = updated_ded_scores.get(code, {}).get(
                            "similarity_scores", []
                        )
                        for i, keyword in enumerate(keywords):
                            new_clusters[keyword]["code_keywords"].append(code)
                            new_clusters[keyword]["similarity_scores"].append(scores[i])
                    st.session_state.clusters = new_clusters

                    updated_table_data = []
                    for code, details in updated_ded_scores.items():
                        keywords = ", ".join(details.get("code_keywords", []))
                        scores = ", ".join(
                            map(str, details.get("similarity_scores", []))
                        )
                        updated_table_data.append([code, keywords, scores])

                    df_ded_updated = pd.DataFrame(
                        updated_table_data,
                        columns=["Code", "Keywords", "Similarity Scores"],
                    )
                    st.session_state.deductive_df = df_ded_updated.copy()
                    st.session_state.deductive_edit_buffer = df_ded_updated.copy()
                    st.session_state.deductive_changes_saved = True
            else:
                st.data_editor(
                    st.session_state.deductive_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    disabled=True,
                    key="deductive_saved_view",
                )

if st.session_state.deductive or st.session_state.inductive:
    st.markdown(
        """<div class="app-container">
    <p class='subtitle'>Step 5: Generate overall text themes</p>
    </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("💡Theme Generation Info", expanded=False):
        st.markdown(
            """<p class='body-app-text'>In this step, the tool generates overarching themes based on the codes and keywords identified in your transcript.</br>
            These themes aim to summarize the main ideas emerging from the text. The generated themes are meant to guide your interpretation, but they are fully editable.</br>
            You can refine, rename, or restructure them directly to better reflect your own analytical perspective or research focus.
            </p>""",
            unsafe_allow_html=True,
        )
    if st.button("Click to see themes in the transcription", key="themes_button"):
        st.session_state.show_themes = True

if st.session_state.show_themes:
    raw_themes = st.session_state.themes
    if not raw_themes:
        themes = []
    elif isinstance(raw_themes, list):
        themes = raw_themes
    else:
        themes = [raw_themes]

    # 1) build a DataFrame
    df_themes = pd.DataFrame({"Themes": themes})
    if (
        "themes_edit_buffer" not in st.session_state or st.session_state.theme_df.empty
    ) and not df_themes.empty:
        st.session_state.theme_df = df_themes.copy()
        st.session_state.themes_edit_buffer = df_themes.copy()
        st.session_state.themes_changes_saved = False
    if (
        "themes_changes_saved" not in st.session_state
        or not st.session_state.themes_changes_saved
    ):

        # 2) display it as a little table
        edited_df_themes = st.data_editor(
            st.session_state.theme_df,
            num_rows="dynamic",
            key="themes_editor",
            use_container_width=True,
        )
        if st.button("Save changes", key="save_themes"):
            st.session_state.themes_edit_buffer = edited_df_themes
            st.session_state.theme_df = edited_df_themes.copy()
            st.session_state.themes = edited_df_themes["Themes"].tolist()
            st.session_state.rerun_pipeline = True
    else:
        st.data_editor(
            st.session_state.theme_df,
            num_rows="dynamic",
            use_container_width=True,
            disabled=True,
            key="themes_saved_view",
        )


if st.session_state.show_themes:
    st.markdown(
        """<div class="app-container">
    <p class='subtitle'>Step 6: Analyze a custom text chunk</p>
    </div>""",
        unsafe_allow_html=True,
    )
    with st.expander("💡Custom Analysis Info", expanded=False):
        st.markdown(
            """<p class='body-app-text'>In this final step, you can select a custom portion of the transcript to analyze in more detail.</br>
            This is useful for zooming in on specific moments, topics, or speakers that you want to explore further.</br>
            Simply copy and paste the segment you'd like to explore. Make sure that the sentences you provide match the original transcript as closely as possible, and that each sentence ends with appropriate punctuation</br>
            (such as a period, question mark, or exclamation mark). The system will extract insights such as keywords, sentiment, and possible codes from the selected chunk.</br>
            </p>""",
            unsafe_allow_html=True,
        )
    if not st.session_state.custom_analysis_mode:
        if st.button(
            "Click to analyze a custom text chunk", key="custom_analysis_button"
        ):
            st.session_state.custom_analysis_mode = True
            st.rerun()


if st.session_state.custom_analysis_mode:
    analysis_choices = []

    if st.session_state.inductive_pipeline_output:
        analysis_choices.append("Inductive coding")
    if st.session_state.deductive_pipeline_output:
        analysis_choices.append("Deductive coding")

    selected_analysis = st.selectbox(
        "Select the analysis result to use for the custom chunk:",
        options=analysis_choices,
        index=0 if len(analysis_choices) > 0 else -1,
        label_visibility="collapsed",
        key="custom_analysis_selectbox",
    )

    input_text = st.text_area(
        "Enter a sentence or a text chunk to analyze:",
        height=100,
        key="custom_text_area",
        label_visibility="hidden",
        placeholder="Enter a sentence or a text chunk to analyze:",
    )
    if st.button("Analyze text chunk"):
        if input_text.strip():
            with st.spinner("Analyzing..."):

                selected_output = (
                    st.session_state.inductive_pipeline_output
                    if selected_analysis == "Inductive coding"
                    else st.session_state.deductive_pipeline_output
                )
                mode = (
                    "inductive"
                    if selected_analysis == "Inductive coding"
                    else "deductive"
                )
                combined_data = custom_pipeline_output(mode)
                custom_analysis_result = analyze_custom_string(
                    input_text,
                    combined_data,
                    models=st.session_state.models,
                    cfg=st.session_state.cfg,
                )
                st.session_state.custom_analysis_result = custom_analysis_result
                if custom_analysis_result is not None:
                    st.markdown(
                        """<p class='steps'>Analysis Result</p>
                    """,
                        unsafe_allow_html=True,
                    )

                    text = custom_analysis_result.get("text", "")
                    keywords = custom_analysis_result.get("keywords", [])
                    highlighted_text = highlight_keywords(text, keywords)
                    st.markdown(
                        f"""
                            <div class="custom-container-speaker">
                                {highlighted_text}
                            </div>

                            """,
                        unsafe_allow_html=True,
                    )

                    sentiment = custom_analysis_result.get("sentiment", {})
                    st.markdown(
                        "<p class='steps'>💬 Sentiment</p>", unsafe_allow_html=True
                    )
                    if sentiment:
                        color = {
                            "very positive": "#38761d",
                            "positive": "#8fce00",
                            "neutral": "#fffa3e",
                            "negative": "#ff8c3e",
                            "very negative": "#F44336",
                        }.get(sentiment["overall_sentiment"].lower(), "#2196F3")

                        st.markdown(
                            f"""
                            <div style="background-color:{color}; padding:10px; border-radius:10px; color:white;">
                                <b>Sentiment:</b> {sentiment['overall_sentiment']} <br>
                                <b>Confidence:</b> {sentiment['confidence']:.2f} <br>
                                <b>Uncertain:</b> {sentiment['uncertain']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("Sentiment not available for this input.")
                    st.markdown(
                        "<p class='steps'>🔍 Keywords and codes</p>",
                        unsafe_allow_html=True,
                    )
                    code_data = [
                        {"Code": code, "Keywords": ", ".join(keywords)}
                        for code, keywords in custom_analysis_result["codes"].items()
                    ]
                    if code_data:
                        df_analysis_codes = pd.DataFrame(code_data)
                        st.data_editor(
                            df_analysis_codes,
                            num_rows="dynamic",
                            use_container_width=True,
                        )

        else:
            st.warning("Please enter a text chunk to analyze.")
