import streamlit as st


def render_transcription(data):
    for i, entry in enumerate(data):
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
