<div align="center">

# ShinAI 🤖

Customizable Telegram bot powered by multiple AI providers with personality, memory, actions, and social context awareness.

[![Chat With The Bot](https://img.shields.io/badge/Chat%20With%20The%20Bot-Telegram-blue.svg?logo=telegram)](https://t.me/shinobi7kbot)

</div>

## Features

- 🧠 **Multiple AI Providers**: Gemini, OpenRouter, Groq, Cerebras, or local LLM (Ollama)
- 💬 **Personality System**: Fully customizable bot personality and behavior
- 🎭 **Social Context**: Recognizes group members and adapts responses
- 📝 **Long-term Memory**: Remembers past conversations using vector embeddings
- 🎨 **Style Learning**: Learns communication patterns from example messages
- 📌 **Sticker Support**: Send stickers as responses with custom mappings
- ⚡ **Rate Limiting**: Built-in cooldowns to prevent spam
- 🔄 **Reply Chain Tracking**: Understands conversation context

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MAymanKH/ShinAI.git
cd ShinAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API credentials
```

### 4. Customize Your Bot

Copy the template files and customize them:

```bash
# Personality configuration
cp shin_ai/data/personality_template.py shin_ai/data/personality.py

# Sticker mappings
cp shin_ai/data/stickers_template.py shin_ai/data/stickers.py

# Group members (optional)
cp shin_ai/data/members_template.py shin_ai/data/members.py
```

### 5. Run the Bot

```bash
python main.py
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_ID` | Telegram API ID from my.telegram.org | ✅ |
| `API_HASH` | Telegram API Hash | ✅ |
| `BOT_TOKEN` | Bot token from @BotFather | ✅ |
| `ADMIN_USER_ID` | Your Telegram user ID | ✅ |
| `AI_CHOICE` | AI provider (gemini/openrouter/groq/cerebras/local/manual) | ✅ |
| `GEMINI_MODEL` | Gemini model name | For Gemini |
| `OPENROUTER_API_KEY` | OpenRouter API key | For OpenRouter |
| `GROQ_API_KEY` | Groq API key | For Groq |
| `CEREBRAS_API_KEY` | Cerebras API key | For Cerebras |

### Personality Configuration

Edit `shin_ai/data/personality.py` to customize:

- `identity`: Who the bot is
- `core_relationships`: Important people/entities
- `behavioral_protocols`: Rules and behaviors
- `interaction_style_personality`: Tone and language style
- `kicking_protocol_trigger_conditions`: When to kick users
- `kicking_protocol_restrictions`: Who cannot be kicked

### Sticker Configuration

Edit `shin_ai/data/stickers.py` to map sticker file IDs to descriptions. Get sticker file IDs by forwarding stickers to @RawDataBot.

### Member Configuration

Edit `shin_ai/data/members.py` to add group members for social context. The bot will recognize them and adapt its responses accordingly.

## Project Structure

```
ShinAI/
├── main.py                 # Entry point
├── shin_ai/
│   ├── bot.py             # Bot initialization
│   ├── config.py          # Configuration loading
│   ├── core/              # Core functionality
│   │   ├── client.py      # Pyrogram client
│   │   ├── prompt_builder.py
│   │   ├── response_parser.py
│   │   ├── action_executor.py
│   │   └── state.py
│   ├── data/              # Data templates
│   │   ├── personality_template.py
│   │   ├── stickers_template.py
│   │   ├── members_template.py
│   │   └── loader.py
│   ├── handlers/          # Message handlers
│   │   ├── chat.py
│   │   ├── fun.py
│   │   └── stats.py
│   ├── providers/         # AI providers
│   │   ├── gemini.py
│   │   ├── openrouter.py
│   │   ├── groq.py
│   │   ├── cerebras.py
│   │   └── local_llm.py
│   ├── services/          # Business logic
│   │   ├── social.py
│   │   └── replies.py
│   ├── stylers/           # Style learning
│   │   ├── style_indexer.py
│   │   └── style_retriever.py
│   └── utils/             # Utilities
│       ├── context_manager.py
│       ├── db.py
│       ├── logger_config.py
│       ├── memory.py
│       └── rate_limit.py
└── data/                  # Runtime data (gitignored)
    ├── gemini_keys.json
    ├── gemini_stats.json
    └── bot_replies.json
```

## Commands

| Command | Description |
|---------|-------------|
| `/gstats` | Show Gemini API key statistics |
| `/gstats_details` | Detailed stats (admin only) |

## Triggers

The bot responds when:
- Mentioned with "يالبوت" (Arabic trigger word)
- Replied to on its previous messages
- Random 1% chance on any group message

---

## How It Works

### Architecture Overview

ShinAI uses a **Retrieval-Augmented Generation (RAG)** architecture to create contextually-aware responses. Rather than relying solely on the AI model's training data, the bot retrieves relevant information from multiple sources before generating a response.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Incoming Message                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Context Collection                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │
│  │ Recent Chat   │  │ Long-term     │  │ Social Context    │    │
│  │ (50 messages) │  │ Memory (RAG)  │  │ (Member Profiles) │    │
│  └───────────────┘  └───────────────┘  └───────────────────┘    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │
│  │ Reply Chain   │  │ Style         │  │ Runtime Metadata  │    │
│  │ Context       │  │ Examples      │  │ (User Status)     │    │
│  └───────────────┘  └───────────────┘  └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    System Prompt Builder                        │
│         Combines personality + context + instructions           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AI Provider (LLM)                          │
│     Gemini │ OpenRouter │ Groq │ Cerebras │ Ollama              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Response Parser                             │
│   Extracts: Text │ Reactions │ Stickers │ Actions │ Targets     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Action Executor                             │
│        Sends messages, reactions, stickers, or kicks            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Deep Dive

### Vector Embeddings & ChromaDB

ShinAI uses **vector embeddings** to enable semantic search across memories and member profiles. Here's how it works:

1. **Text → Vector**: Text is converted into high-dimensional vectors (embeddings) using `sentence-transformers` with the `intfloat/multilingual-e5-large` model
2. **Semantic Similarity**: Similar concepts cluster together in vector space, enabling "meaning-based" search rather than keyword matching
3. **ChromaDB Storage**: Vectors are stored in ChromaDB, a lightweight vector database optimized for embedding search

```python
# Example: "Who created you?" matches the creator's profile
# even without exact keyword matches like "father" or "creator"
query = "Who made this bot?"
# → Semantically matches member with role "Bot creator"
```

### RAG (Retrieval-Augmented Generation)

The bot uses RAG in three key areas:

| Component | What's Retrieved | How It's Used |
|-----------|------------------|---------------|
| **Long-term Memory** | Past conversations with the user | Provides continuity ("Remember when you said...") |
| **Social Context** | Member profiles matching the conversation | Injects relationship info ("This is your creator") |
| **Style Examples** | Similar past responses | Helps maintain consistent voice/tone |

### Memory System

Every interaction is saved to the vector database:

```
[2026-01-30 14:30:00 UTC]
User (@username) said: What's your favorite anime?
Bot replied: steins gate obviously, are you even asking?
```

When a relevant topic comes up later, this memory is retrieved and injected into the prompt, giving the AI context about past interactions.

---

## Bot Capabilities

### 🎭 Group Chat Member Persona

Unlike typical "assistant" bots, ShinAI can act like a **real group member** (depending on your personality configuration):

- **No "How can I help you?"** – Responds naturally without formal greetings
- **Random Interjections** – Sometimes jumps into conversations uninvited (1% chance)
- **Matches Dialect** – Adapts to the group's language style (e.g., Egyptian Arabic slang)
- **Sloppy Typing** – Types like a casual chatter (no punctuation, lowercase, lazy spelling)
- **Teasing & Sarcasm** – Can roast users

### 💬 Smart Response Types

The AI chooses the most appropriate response format:

| Response Type | Format | Example |
|---------------|--------|---------|
| **Text** | Plain message | `"lol nice one"` |
| **Reaction** | `react:<emoji>` | `react:🔥` → Adds 🔥 reaction |
| **Sticker** | `sticker:<file_id>` | Sends a sticker from the configured library |
| **Action** | `action:kick` | Kicks a user (with restrictions) |

### 🎯 Intelligent Reply Targeting

When responding in a reply chain, the bot can choose who to reply to:

```
target:sender      → Reply to the person who triggered the bot
target:parent      → Reply to the message being replied to
target:grandparent → Reply to the message before that
```

This enables conversations like:
> **User A**: "Tell him he's wrong"  
> **Bot**: *replies directly to User B* "you're wrong lol"

### 📌 Sticker Integration

The bot has access to a custom sticker library with semantic descriptions:

```python
STICKER_MAPPINGS = {
    "Laughing": "<file_id>",      # For funny moments
    "Confused": "<file_id>",      # When puzzled
    "Annoyed": "<file_id>",       # When irritated
    "Deal": "<file_id>",          # Handshake/agreement
    # ... more stickers
}
```

The AI selects stickers based on emotional context, not just keywords.

### 😀 Telegram Reactions

Instead of cluttering chat with text, the bot can react with emojis:

- 👍 👎 ❤️ 🔥 😢 🤮 🤯 👀

Reactions are preferred when:
- Acknowledging a message without adding new content
- Responding to stickers (sticker → reaction only)
- Ending a conversation naturally

### 👢 Kick Protocol

The bot can kick users, but with strict safeguards:

**Triggers:**
- Direct order from the creator/admin
- Self-defense (user is extremely annoying)

**Restrictions:**
- Cannot kick admins or owners
- Cannot kick protected users (creator, etc.)
- Ignores kick requests from random users

```
User: "kick him"
Bot: *ignores* (random users can't command kicks)

Creator: "kick @spammer"
Bot: *kicks @spammer*
```

### 🧠 Context Awareness

The bot maintains awareness of:

| Context Type | Window | Purpose |
|--------------|--------|---------|
| **Recent Messages** | Last 50 messages | Understand ongoing conversation |
| **Reply Chain** | Up to 10 levels deep | Follow threaded discussions |
| **User Status** | Real-time | Know if user is admin/owner |
| **Interaction Type** | Per-message | Direct mention vs. random interjection |

### 🔄 Loop Prevention

The bot avoids awkward endless conversations:

- Detects natural endings ("thanks", "ok", "bye", laughing)
- Responds with reaction/sticker instead of forcing more text
- Doesn't reply to its own messages

---

## AI Provider Details

### Gemini (Recommended)

- Supports **image understanding** (photos, stickers)
- Has **Google Search** integration for real-time info
- Multiple API key rotation for quota management
- Tracks key health with `/gstats` command

### OpenRouter

- Access to multiple models (Claude, GPT-4, Llama, etc.)
- Pay-per-token pricing
- Good fallback option

### Groq & Cerebras

- Extremely fast inference
- Good for high-traffic groups
- Limited context windows

### Local LLM (Ollama)

- Fully private, no API costs
- Requires local GPU
- Configure with `LOCAL_MODEL` env var