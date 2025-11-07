from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style, unmatched consistency, and passion for the game.",
    "Lionel Messi is an Argentine professional footballer who plays as a forward and is widely regarded as one of the greatest football players of all time.",
    "The Great Wall of China is a series of fortifications that stretch across northern China, built to protect against invasions and raids from nomadic groups.",
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
    "The Amazon rainforest, often referred to as the 'lungs of the Earth,' is the largest tropical rainforest in the world, home to an incredible diversity of flora and fauna."
]

query = "Who is captain of Argentina football team?"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding],doc_embeddings)[0]

print(sorted(list(
            enumerate(similarities)),
            key=lambda x:x[1],reverse=True))
