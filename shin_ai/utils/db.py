import chromadb
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../chroma_db")

client = chromadb.PersistentClient(path=CHROMA_PATH)

# print(client.get_or_create_collection("style_group").count())