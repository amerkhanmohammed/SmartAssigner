import streamlit as st
from smartassigner2 import app # import your pipeline

st.title("Triagent - AI Powered ticket Triaging")

if st.button("Auto Assign Task"):
    with st.spinner("Running pipeline..."):
        final_state = app.invoke({})
    st.success("Agent finished!")
    st.json(final_state)