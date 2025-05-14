import streamlit as st
import pages.utils.nav as n

st.set_page_config(
    page_title="Thematico",
    page_icon=":bar_chart:",
    layout="wide",
)

n.run(index=0)  # Run navigation logic


lottie1_html = """
<script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
<div style="display: flex; justify-content: flex-start; margin-left: 80px; margin-top: 40px;">
<dotlottie-player
  src="https://lottie.host/16058c46-29ad-4e93-af1c-e048c6ee3747/16zwet7YE6.lottie"
  background="transparent"
  speed="1"
  style="width: 300px;
  height: 300px"
  loop
  autoplay>
</dotlottie-player>
</div>
"""

lottie2_html = """
<script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
<div style="display: flex; justify-content: flex-start; margin-left: 60px; margin-top: 20px;">
<dotlottie-player
  src="https://lottie.host/ca5d40d4-16f6-49f8-8f07-297c03320540/pNRrAWPpdQ.lottie"
  background="transparent"
  speed="1"
  style="width: 100%;
  max-width: 300px;
  height: 300px"
  loop
  autoplay>
</dotlottie-player>
</div>
"""

lottie3_html = """
<script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
<div style="display: flex; justify-content: flex-start; margin-left: 80px; margin-top: 20px;">
<dotlottie-player
  src="https://lottie.host/c538a723-de78-427d-8e11-1c6a80d9a8ab/Vs61qWxcW3.lottie"
  background="transparent"
  speed="1"
  style="width: 100%;
  max-width: 300px;
  height: 300px"
  loop
  autoplay>
</dotlottie-player>
</div>
"""


st.markdown(
    """
  <link href="https://fonts.googleapis.com/css2?family=Acme&display=swap" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
  <style>
  [data-testid="stAppViewContainer"]{
    background-color: #F3E9DC;
  }

  .custom-font-title {
  font-family: "Acme", sans-serif;
  font-weight: 400;
  font-style: normal;
}
  .custom-font-body {
  font-family: "Roboto", sans-serif;
  font-optical-sizing: auto;
  font-weight: 400;
  font-style: normal;
  font-variation-settings: "wdth" 100;
}
  .title{
    text-align: center;
    font-size: 100px;
    font-weight: bold;
    font-family: 'Acme';
    color: black;
    margin-top: -60px;
  }
  .subtitle{
    text-align: center;
    font-size: 30px !important;
    margin-top: -20px;
    font-family: 'Acme';
    color: black;
  }
  .body-home{
    text-align: center;
    font-size: 20px;
    font-family: 'Roboto', sans-serif;
    color: black;
    margin-top: 20px;
  }
  .body-container{
    background-color: #DAB49D;
    padding: 5px;
    border-radius: 20px;
    text-align: center;
    margin: 0px auto;
    width: 100%;
    margin-top: -40px;
  }
  .btn-theme{
    background-color: #DAB49D;
    padding: 0.75em 1.5em;
    border-radius: 10px;
    color: black;
    font-size: 20px;
    font-family: 'Acme', sans-serif;
    cursor: pointer;
    transition: background-color 0.3s ease;
    display: block;
    margin: 2rem auto 0;
  }
  .btn-theme:hover{
    background-color: #c49a7e;
  }
  </style>

  """,
    unsafe_allow_html=True,
)


st.markdown('<div class="title">Thematico</div>', unsafe_allow_html=True)

st.markdown(
    "<p class='subtitle'>Your AI helper for qualitative analysis.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:

    st.components.v1.html(lottie1_html, height=350)
    st.markdown(
        """<div class="body-container">
              <p class='body-home'>
              Are you a researcher who is tired of spending countless hours analyzing
              transcripts of interviews or focus groups?
              Is manual transcript analysis holding back your research?
              </p>
          </div>""",
        unsafe_allow_html=True,
    )


with col2:
    st.components.v1.html(lottie2_html, height=350)
    st.markdown(
        """<div class="body-container">
        <p class='body-home'>
        Are you frustrated by how long it takes to analyze your interview data?
        Are you manually coding qualitative data and finding it overwhelming?
        </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


with col3:
    st.components.v1.html(lottie3_html, height=350)
    st.markdown(
        """<div class="body-container">
        <p class='body-home'>
        Do you wish you could spend less time digging through transcripts and more time actually making sense of what your data is telling you?
        </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """<div style="text-align: center; margin-top: 50px;">""",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>You are in the right place!</p>", unsafe_allow_html=True
)

st.markdown(
    """</div>""",
    unsafe_allow_html=True,
)
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 2])
with col2:
    st.markdown(
        """<div style="
      display: flex;
      justify-content: center;
      align-items: center;
      text-align: center;
      ">
          <a href="/about" target="_self">
              <button class="btn-theme">Learn More about Thematico</button>
          </a>
          </div>
          """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """<div style="
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        ">
            <a href="/application" target="_self">
                <button class="btn-theme">Try out the analytic tool</button>
            </a>
            </div>
            """,
        unsafe_allow_html=True,
    )
