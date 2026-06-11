import uuid
import time
import asyncio
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from shin_ai.utils.db import client
from shin_ai.stylers.style_retriever import embedder
from shin_ai.utils.logger_config import logger
from shin_ai.utils.memory_time import detect_time_filter

# Create the collection for chat memories
memory_collection = client.get_or_create_collection("chat_memories")


# Memory Storage
async def save_memory(platform: str, user_id: int | str, username: str, prompt: str, response: str, context: str = "", chat_id: int | str = 0, chat_title: str = ""):
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

        # Add timestamp and chat title to the readable memory text
        chat_prefix = f" [Chat: {chat_title} on {platform.title()}]" if chat_title else f" [Platform: {platform.title()}]"
        memory_text = f"[{now_str}]{chat_prefix}\n{memory_text}"

        # Metadata for filtering/context
        meta = {
            "platform": platform,
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
        # We include the timestamp and chat_title in the passage so it's nominally searchable.
        searchable_text = f"passage: [{now_str}]{chat_prefix} User ({username}) said: {prompt}\nBot replied: {response}"
        # Off-thread to avoid blocking event loop
        embedding_tensor = await asyncio.to_thread(embedder.encode, searchable_text)
        embedding = embedding_tensor.tolist()
        
        memory_collection.add(
            ids=[mem_id],
            documents=[memory_text],
            embeddings=[embedding],
            metadatas=[meta]
        )
        logger.debug("Memory saved for user %s (chat=%s platform=%s)", username, chat_id, platform)
    except Exception as e:
        logger.error("Failed to save memory for user %s: %s", username, e, exc_info=True)


# Memory Retrieval

def _apply_mmr(query_emb: list, candidate_docs: list, candidate_embs: list, limit: int, lambda_param: float = 0.65) -> list:
    """
    Maximal Marginal Relevance (MMR) for diverse retrieval.
    Selects documents that are highly relevant but mutually diverse.
    """
    if not candidate_docs:
        return []
        
    query_tensor = np.array(query_emb).reshape(1, -1)
    cand_tensor = np.array(candidate_embs)
    
    # Calculate similarity between query and all candidates
    sim_to_query = cosine_similarity(query_tensor, cand_tensor)[0]
    
    selected_indices = []
    available_indices = list(range(len(candidate_docs)))
    
    # Pre-calculate similarity between all candidates for speed
    cand_sim_matrix = cosine_similarity(cand_tensor)
    
    while len(selected_indices) < limit and available_indices:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in available_indices:
            rel_score = sim_to_query[idx]
            
            # Diversity penalty: max similarity to already selected docs
            if selected_indices:
                div_score = max([cand_sim_matrix[idx][s_idx] for s_idx in selected_indices])
            else:
                div_score = 0.0
                
            # MMR formula
            mmr_score = lambda_param * rel_score - (1.0 - lambda_param) * div_score
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
                
        if best_idx != -1:
            selected_indices.append(best_idx)
            available_indices.remove(best_idx)
        else:
            break
            
    return [candidate_docs[i] for i in selected_indices]


async def retrieve_memories(query: str, limit: int = 15):
    """
    Retrieves semantically relevant past interactions.
    If the query contains a time reference (e.g. "2 days ago", "قبل ساعة"),
    results are constrained to that time window via ChromaDB metadata filtering.
    """
    try:
        # E5 requires "query: " prefix for search queries, off-thread
        query_emb_tensor = await asyncio.to_thread(embedder.encode, f"query: {query}")
        query_emb = query_emb_tensor.tolist()

        # Check for time references in the query
        start_epoch, end_epoch = await detect_time_filter(query)

        where_filter = None
        if start_epoch is not None and end_epoch is not None:
            where_filter = {
                "$and": [
                    {"timestamp": {"$gte": start_epoch}},
                    {"timestamp": {"$lte": end_epoch}},
                ]
            }
            logger.debug("Time-filtered memory search: %s → %s", start_epoch, end_epoch)

        # Fetch a large pool for MMR deduplication
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=40,
            where=where_filter,
            include=["documents", "distances", "embeddings"]
        )
        
        filtered_docs = []
        filtered_embs = []
        
        if results['documents']:
            docs = results['documents'][0]
            dists = results['distances'][0]
            embs = results['embeddings'][0]

            # Use a more lenient threshold when time-filtering
            threshold = 1.5 if where_filter else 1.3

            for doc, dist, emb in zip(docs, dists, embs):
                if dist < threshold:
                    filtered_docs.append(doc)
                    filtered_embs.append(emb)

        # Apply MMR Deduplication
        final_memories = _apply_mmr(query_emb, filtered_docs, filtered_embs, limit)

        # If time filter was applied but returned nothing, fall back to unfiltered
        if where_filter and not final_memories:
            logger.debug("Time-filtered search returned no results — falling back to unfiltered")
            return await _retrieve_memories_unfiltered(query_emb, limit)

        return final_memories
    except Exception as e:
        logger.error("Failed to retrieve memories: %s", e, exc_info=True)
        return []


async def _retrieve_memories_unfiltered(query_emb: list, limit: int = 15):
    """
    Fallback: pure semantic retrieval without any time filter.
    Accepts a pre-computed embedding to avoid re-encoding.
    """
    try:
        results = memory_collection.query(
            query_embeddings=[query_emb],
            n_results=40,
            include=["documents", "distances", "embeddings"]
        )
        filtered_docs = []
        filtered_embs = []
        
        if results['documents']:
            for doc, dist, emb in zip(results['documents'][0], results['distances'][0], results['embeddings'][0]):
                if dist < 1.3:
                    filtered_docs.append(doc)
                    filtered_embs.append(emb)
                    
        return _apply_mmr(query_emb, filtered_docs, filtered_embs, limit)
    except Exception as e:
        logger.error("Failed to retrieve memories (unfiltered fallback): %s", e, exc_info=True)
        return []
