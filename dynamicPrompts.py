
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


import urllib3
import os
import requests
# ---------------------------------------------------------------------
# Disable SSL verification globally (for corporate restricted machines)
# ---------------------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from requests import adapters
from urllib3.util.ssl_ import create_urllib3_context

class NoVerifyHTTPAdapter(adapters.HTTPAdapter):
    """Custom adapter that disables SSL certificate verification."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = False
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

# Create a single no-verify session
session = requests.Session()
session.verify = False
adapter = NoVerifyHTTPAdapter()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Patch Hugging Face internals to use this session
try:
    import huggingface_hub.utils as hf_utils
    def patched_get_session():
        return session
    hf_utils.get_session = patched_get_session
    print("[INFO] SSL verification bypass active (corporate network patch applied).")
except Exception as e:
    print(f"[WARN] Could not patch huggingface_hub session: {e}")

# ---------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------
load_dotenv()
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------------------------------------------------------------
# HuggingFaceEndpoint wrapper
# ---------------------------------------------------------------------
class NoVerifyHuggingFaceEndpoint(HuggingFaceEndpoint):
    """Extends HuggingFaceEndpoint to use the no-verify session explicitly."""
    def _send_request(self, url, headers, payload):
        response = session.post(
            url,
            headers=headers,
            json=payload,
            verify=False,   # SSL bypassed here as well
            timeout=60
        )
        response.raise_for_status()
        return response.json()

try:
    llm = NoVerifyHuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=hf_key,
    )
    model = ChatHuggingFace(llm=llm)
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🧠 Research Tool (SSL bypass mode)")

user_input = st.text_area("Enter your query:")

if st.button("Generate Response"):
    if not user_input.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(user_input)
                st.success("Response:")
                st.write(result.content)
            except Exception as e:
                st.error(f"⚠️ Error during generation: {e}")
