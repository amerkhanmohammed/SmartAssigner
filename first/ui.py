import streamlit as st
import io, sys
from smartassigner2 import app

st.title("Triagent - AI Powered ticket Triaging")

if st.button("Auto Assign the Open Tasks"):
    log_placeholder = st.empty()
    buffer = io.StringIO()

    class StreamToUI(io.TextIOBase):
        def write(self, s):
            buffer.write(s)
            log_placeholder.text(buffer.getvalue())  # show logs live
            return len(s)

    old_stdout = sys.stdout   # save original stdout
    try:
        sys.stdout = StreamToUI()   # redirect prints
        final_state = app.invoke({})
    finally:
        sys.stdout = old_stdout     # restore terminal output ✅

    st.success("Agent finished!")