import os

import pages.utils.styles as styles
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page


def run(index):
    styles.apply_styles()

    st.markdown(
        """<style> .css-79elbk {display:none}</style>""", unsafe_allow_html=True
    )
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"] {
        background-color: #895737 !important;
    }
    .css-1d391kg{
        padding: 20 !important;
    }
    .nav-container{
        border: ;
        box-shadow: none !important;
    }
    .css-1v3fvcr, .css-79elbk, .css-1d391kg {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        options_arr = ["Home", "About", "Application"]
        choose = option_menu(
            "Navigation",
            options_arr,
            icons=["house", "info-circle", "robot"],
            menu_icon="app-indicator",
            default_index=index,
            styles={
                "container": {"padding": "5!important", "background-color": "#c08552"},
                "icon": {"color": "black", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#dab49d",
                    "color": "#000000",
                },
                "nav-link-selected": {
                    "background-color": "#dab49d",
                    "color": "#000000",
                },
            },
        )

        for c_idx, option in enumerate(options_arr):
            if option == choose and index != c_idx:
                switch_page(choose.lower())
