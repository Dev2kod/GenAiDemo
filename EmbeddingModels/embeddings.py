from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "Yash"
docs = [
    "Leo Messi",
    "Lionel Messi",
    "Lionel Andres Messi Cuccittini",
    "Messi"
]
vector = embedding.embed_query(text)
# vector2 = embedding.embed_documents(docs)
print(str(vector))
