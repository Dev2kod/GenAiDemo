from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

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