import uuid
import time
from datetime import datetime
from shin_ai.utils.db import client
from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger

# Create the collection for chat memories
memory_collection = client.get_or_create_collection("chat_memories")

def save_memory(user_id: int, username: str, prompt: str, response: str, context: str = "", chat_id: int = 0, chat_title: str = ""):
    """
    Saves a user-bot interaction to the vector database.
    """
    try:
        if not response or not prompt:
            return
        
        # Get formatted timestamp
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        # Format the memory text. 
        if context:
            # If there is context (previous reply), include it so the memory stands on its own
            memory_text = f"Context: {context}\nUser ({username}) said: {prompt}\nBot replied: {response}"
        else:
            memory_text = f"User ({username}) said: {prompt}\nBot replied: {response}"
        
        # Clean up reaction responses for better reading in future
        if response.startswith("react:"):
            reaction = response.split(":")[1]
            memory_text = f"User ({username}) said: {prompt}\nBot reacted with: {reaction}"
        elif response.startswith("sticker:"):
            memory_text = f"User ({username}) said: {prompt}\nBot sent a sticker."

        # Add timestamp to the readable memory text
        memory_text = f"[{now_str}]\n{memory_text}"

        # Metadata for filtering/context
        meta = {
            "user_id": str(user_id),
            "username": username or "Unknown",
            "timestamp": int(time.time()),
            "date_string": now_str,
            "type": "conversation"
        }
        if chat_id:
            meta["chat_id"] = str(chat_id)
        if chat_title:
            meta["chat_title"] = chat_title
        
        # Unique Memory ID
        mem_id = str(uuid.uuid4())
        
        # Create embedding
        # We specifically embed the interaction itself, ignoring the previous context prefix
        # This ensures that searching for "What did I say?" matches the actual content, not the context noise.
        # E5 requires "passage: " prefix for documents to be stored
        # We include the timestamp in the passage so it's nominally searchable, though semantic match is primary.
        searchable_text = f"passage: [{now_str}] User ({username}) said: {prompt}\nBot replied: {response}"
        embedding = embedder.encode(searchable_text).tolist()
        
        memory_collection.add(
            ids=[mem_id],
            documents=[memory_text],
            embeddings=[embedding],
            metadatas=[meta]
        )
        logger.info(f"Memory saved for user {username}")
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")

def retrieve_memories(query: str, limit: int = 5):
    """
    Retrieves semantically relevant past interactions.
    """
    try:
        # E5 requires "query: " prefix for search queries
        query_emb = embedder.encode(f"query: {query}").tolist()
        
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=limit,
            include=["documents", "distances"]
        )
        
        filtered_memories = []
        if results['documents']:
            docs = results['documents'][0]
            dists = results['distances'][0]
            
            for doc, dist in zip(docs, dists):
                # Filter out irrelevant memories (Distance threshold)
                # Lower distance = higher similarity. 
                # 1.3 is a balanced threshold for sentence-transformers L2 distance.
                if dist < 1.3:
                    filtered_memories.append(doc)
                    
            return filtered_memories
        return []
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}")
        return []
