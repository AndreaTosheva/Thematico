import streamlit as st
import pandas as pd
import numpy as np
import requests

import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import os

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page


import streamlit as st
import pages.utils.nav as n


def apply_styles():

    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Acme&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
    <style>
    [data-testid="stAppViewContainer"]{
        background-color: #F3E9DC;
    }
    [data-testid="stTextArea"]{
        margin-top: -35px !important;
    }
    [data-testid="stExpander"]{
        color: black !important;
        font-style: italic !important;
    }
    [data-testid="stExpander"] > div > div{
        color: black !important;
        font-style: italic !important;
    }

    [data-testid="stFileUploader"]{
        color: black !important;
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
    .steps{
        text-align: left;
        font-size: 25px !important;
        font-family: 'Acme';
        color: black;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .body{
        text-align: center;
        font-size: 20px;
        font-family: 'Roboto', sans-serif;
        color: black;
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
    .app-container{
        background-color: #f3e9dc;
        padding: 5px;
        border-radius: 20px;
        display: inline-block;
        text-align: left;
        margin: 0 auto;
        margin-top: -10px;
        margin-right: 10px;
        width: 100%;
    }
    .body-app-text{
        text-align: left;
        font-size: 20px;
        font-family: 'Roboto', sans-serif;
        color: black;
    }
    .body-app-text-keywords{
        text-align: left;
        font-size: 15px;
        font-family: 'Roboto', sans-serif;
        color: black;
        font-weight: bold;
    }
    .custom-container-speaker {
        background-color: #f3e9dc;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .custom-textbox textarea {
        background-color: #1e1e28;
        color: white;
        border: 2px solid #c08552;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    .main-container-app{
        max-width: 800px;
        margin: auto;
    }
    </style>

    """,
        unsafe_allow_html=True,
    )
