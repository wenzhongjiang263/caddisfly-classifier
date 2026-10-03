"""Streamlit Community Cloud entry point; homepage stays independent of models."""

import streamlit as st

import app


st.session_state.cloud_demo = True
app.main()
