import chromadb

from shin_ai.config import CHROMA_PATH

client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# print(client.get_or_create_collection("style_group").count())
