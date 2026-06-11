from sentence_transformers import SentenceTransformer
from shin_ai.utils.db import client
from shin_ai.config import EMBEDDING_MODEL

collection = client.get_or_create_collection("style_group")
embedder = SentenceTransformer(EMBEDDING_MODEL)


def get_style_examples(query, k=10):
    q_emb = embedder.encode(f"query: {query}").tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]
