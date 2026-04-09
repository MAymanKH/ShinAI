"""
Social Context Service

Manages semantic retrieval of member information for contextual responses.
"""
import re
from shin_ai.platforms.models import UnifiedMessage
from shin_ai.utils.db import client
from shin_ai.utils.logger_config import logger
from shin_ai.stylers.style_retriever import embedder
from shin_ai.data.loader import MEMBERS

# --- SEMANTIC SOCIAL CONTEXT SETUP ---
# We delete and recreate the collection to ensure that IDs removed from MEMBERS code are also removed from DB
try:
    client.delete_collection("social_context_members")
except:
    pass
social_collection = client.get_or_create_collection("social_context_members")


def _username_field_for_platform(platform: str) -> str:
    """Returns the member dict key holding the username for a given platform."""
    if platform == "discord":
        return "discord_username"
    if platform == "whatsapp":
        return "whatsapp_username"
    return "telegram_username"


def resolve_username_to_key(username: str, platform: str = "") -> str | None:
    """
    Resolve a platform username to a MEMBERS dict key.
    
    If platform is provided, check the platform-specific username field first.
    Falls back to checking both fields and the names list.
    """
    clean = username.lower().strip().lstrip("@")
    
    # 1. Platform-specific field (prioritised)
    if platform:
        field = _username_field_for_platform(platform)
        for key, data in MEMBERS.items():
            if data.get(field, "").lower() == clean:
                return key

    # 2. Check both platform fields
    for key, data in MEMBERS.items():
        if data.get("telegram_username", "").lower() == clean:
            return key
        if data.get("discord_username", "").lower() == clean:
            return key
        if data.get("whatsapp_username", "").lower() == clean:
            return key

    # 3. Fallback: names list and dict key
    for key, data in MEMBERS.items():
        if clean in [n.lower().lstrip("@") for n in data.get("names", [])]:
            return key
        if key.lower() == clean:
            return key

    return None


def get_platform_username_for_member(member_key: str, platform: str) -> str | None:
    """Given a member key, return the username they use on the specified platform."""
    data = MEMBERS.get(member_key)
    if not data:
        return None
    field = _username_field_for_platform(platform)
    username = data.get(field, "")
    return username if username else None


def index_social_context():
    """Indexes members into ChromaDB for semantic retrieval."""
    ids = []
    documents = []
    metadatas = []
    
    for key, data in MEMBERS.items():
        # Create a rich text representation for semantic matching
        # blending keywords, role, names, usernames, and backstory
        keywords = " ".join(data.get("trigger_keywords", []))
        names = " ".join(data.get("names", []))
        tg_user = data.get("telegram_username", "")
        dc_user = data.get("discord_username", "")
        wa_user = data.get("whatsapp_username", "")
        
        # We index the 'meaning' of the person relative to the bot
        text = f"{names} {tg_user} {dc_user} {wa_user} {data['preferred_name']} {data['role']} {keywords} {data.get('backstory', '')}"
        
        ids.append(key)
        documents.append(text)
        metadatas.append({"preferred_name": data["preferred_name"]})

    # Upsert (update or insert)
    if ids:
        # E5 requires "passage: " prefix for stored documents
        prefixed_documents = [f"passage: {doc}" for doc in documents]
        keywords_embeddings = embedder.encode(prefixed_documents).tolist()
        social_collection.upsert(ids=ids, embeddings=keywords_embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"Indexed {len(ids)} members for social context.")

def get_social_context(msg: UnifiedMessage, reply_chain_text: str = "") -> str:
    """
    Analyzes the message and reply chain to decide which members' lore to inject.
    Returns a string containing the relevant social context.
    """
    platform = msg.platform
    
    active_keys = set()
    sender_key = None
    target_key = None
    
    # 1. Add the Sender & Identify (platform-aware)
    if msg.from_user:
        if msg.from_user.username:
            resolved = resolve_username_to_key(msg.from_user.username, platform)
            if resolved:
                active_keys.add(resolved)
                sender_key = resolved
        
        # Fallback Name Check if username didn't match
        if not sender_key and msg.from_user.first_name:
            fname = msg.from_user.first_name.lower().strip()
            if listbox := [k for k, v in MEMBERS.items() if fname in [n.lower() for n in v['names']]]:
                active_keys.add(listbox[0])
                sender_key = listbox[0]

    # 2. Add the Reply Target (if any) & Identify (platform-aware)
    if msg.reply_to_message and msg.reply_to_message.from_user:
        if msg.reply_to_message.from_user.username:
            resolved = resolve_username_to_key(msg.reply_to_message.from_user.username, platform)
            if resolved:
                active_keys.add(resolved)
                target_key = resolved
        
        if not target_key and msg.reply_to_message.from_user.first_name:
            fname = msg.reply_to_message.from_user.first_name.lower().strip()
            if listbox := [k for k, v in MEMBERS.items() if fname in [n.lower() for n in v['names']]]:
                active_keys.add(listbox[0])
                target_key = listbox[0]

    # 3. Scan text for mentions (Exact Match)
    combined_text = (msg.text or msg.caption or "") + " " + reply_chain_text
    
    # Check all names/aliases for explicit mentions
    for key, data in MEMBERS.items():
        if key in active_keys: continue
        for name in data['names']:
            if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", combined_text.lower()):
                active_keys.add(key)
                break

    # 4. Semantic Search (The Smart Trigger)
    try:
        # We query using the message content to find conceptually related members
        # e.g. "Who is your creator?" -> matches "creator" in MaymanKH's doc
        # E5 requires "query: " prefix
        query_emb = embedder.encode(f"query: {combined_text}").tolist()
        
        # We fetch up to 2 most relevant members
        results = social_collection.query(
            query_embeddings=[query_emb],
            n_results=2, 
            include=["distances"]
        )
        
        if results['ids']:
            ids = results['ids'][0]
            dists = results['distances'][0]
            
            for i, member_id in enumerate(ids):
                if dists[i] < 1.1: 
                    active_keys.add(member_id)
                    
    except Exception as e:
        logger.error(f"Semantic context search failed: {e}")

    if not active_keys:
        return ""

    # Build the output string
    context_lines = ["SOCIAL CONTEXT (RELEVANT MEMBERS PRESENT):"]
    for key in active_keys:
        if key not in MEMBERS:
            continue
        member = MEMBERS[key]
        
        # Add aliases to context so the bot knows who is who
        aliases = [n for n in member['names'] if n.lower() not in member['preferred_name'].lower()]
        aliases_str = f" (aka: {', '.join(aliases)})" if aliases else ""
        
        line = f"- {member['preferred_name']}{aliases_str}: {member['role']}"
        if member.get("location"):
            line += f". Location: {member['location']}"
        if member.get("backstory"):
            line += f". Backstory: {member['backstory']}"
        context_lines.append(line)
    
    # Explicit Identification Section
    context_lines.append("\nCURRENT INTERACTION IDENTITIES (STRICT MAPPING):")
    if sender_key and sender_key in MEMBERS:
        s_mem = MEMBERS[sender_key]
        context_lines.append(f"THE USER TALKING TO YOU ({msg.from_user.first_name} / @{msg.from_user.username or 'NoUser'}) IS: {s_mem['preferred_name']}.")
    else:
        context_lines.append(f"THE USER TALKING TO YOU ({msg.from_user.first_name}) is Unknown (Guest).")

    if target_key and target_key in MEMBERS:
        t_mem = MEMBERS[target_key]
        if msg.reply_to_message:
            context_lines.append(f"THE USER BEING REPLIED TO ({msg.reply_to_message.from_user.first_name}) IS: {t_mem['preferred_name']}.")
    
    context_lines.append("\nSTRICT RULES FOR CONTEXT:")
    context_lines.append("1. NAMING: You MUST address the members above using ONLY their 'preferred_name' (e.g. use '{preferred_name}' instead of @username).")
    context_lines.append("2. BACKSTORY & LOCATION: The 'Backstory' and 'Location' fields are HIDDEN KNOWLEDGE. DO NOT mention them unless the conversation TOPIC explicitly requires it (e.g. asking 'where are you from?' or 'what do you study?'). If we are just joking around, DO NOT bring up their university or history. Keep it natural.")
    
    return "\n".join(context_lines)
