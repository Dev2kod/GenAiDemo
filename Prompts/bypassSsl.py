import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import urllib3
import os
import requests
from requests import adapters
from urllib3.util.ssl_ import create_urllib3_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def bypassSsl():
    class NoVerifyHTTPAdapter(adapters.HTTPAdapter):
        """Custom adapter that disables SSL certificate verification."""
        def init_poolmanager(self, *args, **kwargs):
            ctx = create_urllib3_context()
            ctx.check_hostname = False
            ctx.verify_mode = False
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)
# ---------------------------------------------------------------------
# Disable SSL verification globally (for corporate restricted machines)
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
load_dotenv()
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# ---------------------------------------------------------------------

# HuggingFaceEndpoint wrapper
class HuggingFaceEndpointNoSSL(HuggingFaceEndpoint):
    """HuggingFaceEndpoint wrapper that uses a no-SSL session."""

    def _get_request_kwargs(self) -> dict:
        """Get the request kwargs, overriding the session to disable SSL verification."""
        kwargs = super()._get_request_kwargs()
        kwargs["session"] = session  # Use the no-SSL session
        return kwargs
class NoVerifyHuggingFaceEndpoint(HuggingFaceEndpointNoSSL):
    """HuggingFaceEndpoint wrapper that disables SSL verification."""
    pass
    """HuggingFaceEndpoint wrapper that disables SSL verification."""
    pass
# ---------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    endpoint_url = "https://api-inference.huggingface.co/models/gpt2"
    hf_endpoint = NoVerifyHuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        headers={"Authorization": f"Bearer {hf_key}"},
    )
    llm = ChatHuggingFace(endpoint=hf_endpoint)

    prompt = "Hello, how are you?"
    response = llm.invoke(prompt)
    print("Response:", response)
# ---------------------------------------------------------------------
# HuggingFaceEndpoint wrapper
class NoVerifyHuggingFaceEndpoint(HuggingFaceEndpoint):
    """Extends HuggingFaceEndpoint to use the no-verify session explicitly."""
    def _send_request(self, url, headers, payload):
        response = session.post(
            url,
            headers=headers,
            json=payload,
        )
        return response     

