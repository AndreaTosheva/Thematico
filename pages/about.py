import streamlit as st
import pages.utils.nav as n
import torch
import os
import sys
import pages.utils.styles as styles

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./thematico"))
sys.path.append(os.path.abspath("./pages/utils"))

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
st.set_page_config(
    page_title="Thematico",
    page_icon=":bar_chart:",
    layout="wide",
)
styles.apply_styles()
n.run(index=1)

st.markdown("<div class='main-container-app'>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>AI-Powered Transcript Analysis</p>", unsafe_allow_html=True
)
st.markdown(
    """<br>
            <br>""",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='body-container'> <p class='body'> <strong>Thematico</strong> is your intelligent assistant for making sense of spoken conversations. Upload any transcribed dialogue — from interviews to podcasts — and let Thematico guide you through keyword extraction, sentiment interpretation, thematic coding, and insight discovery.</p></div>",
    unsafe_allow_html=True,
)
st.markdown("<br><p class='steps'>💡 In-App Hints</p>", unsafe_allow_html=True)
st.markdown(
    """
<p class='body-app-text'>
Wherever you see the <strong>💡 lightbulb icon</strong>, click it to reveal helpful tips or explanations about how that part of the tool works. These hints are designed to guide you through Thematico’s features — especially if you're exploring it for the first time.
</p>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>🧠 How Thematico Works</p>", unsafe_allow_html=True)
st.markdown(
    """
<p class='body-app-text'>
Thematico combines cutting-edge NLP models and custom logic to deliver intelligent, context-aware analysis. Here's how the system works under the hood:
</p>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<ul class='body-app-text'>
    <li><strong>Preprocessing</strong>: Uploaded transcriptions are validated and parsed to ensure consistent format and speaker segmentation.</li>
    <li><strong>Keyword Extraction</strong>: Leveraging <strong>KeyBERT</strong> with <strong>SentenceTransformer</strong>, the app identifies relevant and diverse keywords based on configurable thresholds, character limits, and model embeddings.</li>
    <li><strong>Sentiment Analysis</strong>: Each utterance is evaluated using a fine-tuned BERT-based sentiment model, delivering confidence scores and uncertainty flags.</li>
    <li><strong>Theme Generation</strong>:
        A prompt-driven method synthesizes themes from keywords using:
        <ul>
            <li><strong>Ollama</strong> for LLM-powered responses via local inference</li>
            <li>Optional fallback to HuggingFace models</li>
        </ul>
    </li>
    <li><strong>Inductive Coding</strong>: Automatically clusters semantically similar keywords into codes using cosine similarity and density-based clustering (HDBSCAN-inspired).</li>
    <li><strong>Deductive Coding</strong>: Matches user-defined codes to relevant keywords using semantic similarity (top-k ranked with thresholds).</li>
    <li><strong>Custom Text Analysis</strong>: Dynamically analyzes user-inputted strings using the same pipeline: extracting keywords, mapping codes, and predicting sentiment.</li>
</ul>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>🧰 Tech Stack</p>", unsafe_allow_html=True)
st.markdown(
    """
<ul class='body-app-text'>
    <li><strong>Language Models</strong>: Ollama, Qwen2.5-7B-Instruct, BERT</li>
    <li><strong>Libraries</strong>: Transformers, SentenceTransformers, KeyBERT, Pydantic</li>
    <li><strong>Frontend</strong>: Streamlit with custom styling and interactivity</li>
    <li><strong>Optional LLM Modes</strong>: Local (Hugging Face) and API-based (Ollama)</li>
</ul>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>🎯 Key Capabilities</p>", unsafe_allow_html=True)
st.markdown(
    """
<ul class='body-app-text'>
    <li><strong>Transcription Input</strong>: Upload files in JSON, TXT, or CSV format.</li>
    <li><strong>Sentiment Analysis</strong>: Understand the emotional tone with confidence scores.</li>
    <li><strong>Keyword Extraction</strong>: Identify important terms using semantic relevance.</li>
    <li><strong>Inductive & Deductive Coding</strong>: Auto-discover or define your own codes.</li>
    <li><strong>Theme Generation</strong>: Group content into coherent themes based on patterns.</li>
    <li><strong>Custom Text Analysis</strong>: Get insights from any user-defined transcript chunk.</li>
    <li><strong>Word Cloud</strong>: Visualize the dominant language of your conversation.</li>
</ul>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>👥 Built For</p>", unsafe_allow_html=True)
st.markdown(
    """
<p class='body-app-text'>
Whether you're a researcher analyzing interviews, a designer conducting user studies, or a marketer decoding focus groups — Thematico streamlines your qualitative analysis pipeline.
</p>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>🤖 Under the Hood</p>", unsafe_allow_html=True)
st.markdown(
    """
<ul class='body-app-text'>
    <li><strong>KeyBERT</strong> – For high-precision keyword extraction</li>
    <li><strong>Sentence Transformers</strong> – For semantic similarity scoring</li>
    <li><strong>Ollama (LLM)</strong> – For rich language understanding</li>
    <li><strong>Streamlit</strong> – For fast, interactive interfaces</li>
</ul>
""",
    unsafe_allow_html=True,
)

st.markdown("<br><p class='steps'>📬 Contact</p>", unsafe_allow_html=True)
st.markdown(
    """
<p class='body-app-text'>
Have feedback or need support? Reach out at <strong>andreatosheva10@gmail.com</strong>.
</p>
""",
    unsafe_allow_html=True,
)
