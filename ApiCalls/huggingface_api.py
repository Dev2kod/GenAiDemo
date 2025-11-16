from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import warnings
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Monkey patch requests to disable SSL verification globally
import requests
from requests import adapters
from urllib3.util.ssl_ import create_urllib3_context

# Create a custom SSL context that doesn't verify
class NoVerifyHTTPAdapter(adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = create_urllib3_context()
        kwargs['ssl_context'].check_hostname = False
        kwargs['ssl_context'].verify_mode = False
        return super().init_poolmanager(*args, **kwargs)

# Mount it to all sessions
session = requests.Session()
session.verify = False
adapter = NoVerifyHTTPAdapter()
session.mount('https://', adapter)
session.mount('http://', adapter)

# Patch the get_session function in huggingface_hub
import huggingface_hub.utils as hf_utils
def patched_get_session():
    return session

hf_utils.get_session = patched_get_session

# Now import and use normally
load_dotenv()
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(f"API Key loaded: {hf_key[:10]}...")

try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=hf_key,
        )

    model = ChatHuggingFace(llm=llm)
    print("reached endpoint")
    result = model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
