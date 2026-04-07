<div align="center">

# ShinAI

An intelligent Telegram bot that acts like a real group member - not an assistant. Features personality-driven responses, long-term memory with RAG architecture, style learning, and contextual awareness across multiple AI providers.

[![Chat With The Bot](https://img.shields.io/badge/Chat%20With%20The%20Bot-Telegram-blue.svg?logo=telegram)](https://t.me/shinobi7kbot)

</div>

## Table of Contents

- [Inspiration](#inspiration)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Personality Configuration](#personality-configuration)
  - [Sticker Configuration](#sticker-configuration)
  - [Member Configuration](#member-configuration)
- [Project Structure](#project-structure)
- [Commands](#commands)
- [Triggers](#triggers)
- [How It Works](#how-it-works)
  - [Architecture Overview](#architecture-overview)
  - [Vector Embeddings & ChromaDB](#vector-embeddings--chromadb)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Memory System](#memory-system)
- [Bot Capabilities](#bot-capabilities)
  - [Group Chat Member Persona](#group-chat-member-persona)
  - [Smart Response Types](#smart-response-types)
  - [Multi-Message Responses](#multi-message-responses)
  - [Intelligent Reply Targeting](#intelligent-reply-targeting)
  - [Reliability & Retry Behavior](#reliability--retry-behavior)
  - [Sticker Integration](#sticker-integration)
  - [Telegram Reactions](#telegram-reactions)
  - [Moderation System](#moderation-system)
  - [Real-Time Web Search](#real-time-web-search)
  - [Context Awareness](#context-awareness)
  - [Loop Prevention](#loop-prevention)
- [AI Provider Details](#ai-provider-details)
- [License](#license)

## Inspiration

I was tired of AI bots that felt like... bots. You know, the type: "As an AI language model, I cannot..." or the overly formal "How can I assist you today?" that kills the vibe in a casual group chat with friends.

I wanted an AI that didn't just stand on the sidelines waiting for a command, but actually **lived** in the group chat. I wanted a "digital homie". Someone who knows the inside jokes, understands the group's specific slang, remembers that embarrassing thing you said three weeks ago, and isn't afraid to roast you for it. And definitely without being so cringe and unbearable. **ShinAI** was born from the desire to bridge the gap between "helpful assistant" and "chaotic group member."

## Features

- 🧠 **Multiple AI Providers**: Gemini, OpenRouter, Groq, Cerebras, or local LLM (Ollama)
- 💬 **Personality System**: Fully customizable bot personality and behavior
- 🎭 **Social Context**: Recognizes group members and adapts responses
- 🔄 **Reply Chain Tracking**: Understands conversation context
- 📝 **Long-term Memory**: Remembers past conversations using vector embeddings
- 🎨 **Style Learning**: Learns communication patterns from example messages
- 🌐 **Real-Time Web Search**: Web searching, fetching, and scraping capabilities used when needed
- 📌 **Sticker Support**: Send stickers as responses with custom mappings
- 😀 **Emoji Reactions**: React to messages with emojis instead of text
- 📨 **Multi-Message Responses**: Send multiple sequential messages with automatic delays
- 🎯 **Smart Reply Targeting**: Target any message in the chat by its real Telegram ID
- 🛡️ **Moderation Suite**: Kick, ban, unban, mute, unmute, and invite users with configurable escalation
- 🔁 **Reliability Layer**: Per-attempt timeout, retries mechanism, and error-aware retry context injection
- ⚡ **Rate Limiting**: Built-in cooldowns to prevent spam

## Quick Start

You can either [chat with the bot](https://t.me/shinobi7kbot) using its [triggers](#triggers) and add it to your group chats, or build it from source yourself:

### 0. Install Prerequisites

- [Python](https://www.python.org/downloads/) 3.7 or higher
- [Git](https://git-scm.com/install/)
- Make sure they are added to your path.

### 1. Clone the Repository

```bash
git clone https://github.com/MAymanKH/ShinAI.git
cd ShinAI
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials and settings
```

### 5. Customize Your Bot

Copy the template files and customize them:

```bash
# Personality configuration
cp shin_ai/data/personality_template.py shin_ai/data/personality.py

# Sticker mappings
cp shin_ai/data/stickers_template.py shin_ai/data/stickers.py

# Group members (optional)
cp shin_ai/data/members_template.py shin_ai/data/members.py
```

### 6. (Optional) Index Style Examples

If you want the bot to learn from a specific group's communication style:

```bash
# 1. Add STYLE_GROUP_ID to your .env (get group ID from @RawDataBot)
# 2. Run the style indexer
python -m shin_ai.stylers.style_indexer
```

### 7. Run the Bot

**Option A: Native installation (Python)**
```bash
python main.py
```

**Option B: Docker**
If you prefer to use Docker and Docker Compose, make sure they are installed on your system.

```bash
# 1. First time setup: create empty session files so Docker maps them as files, not directories
touch shin_ai_bot.session shin_ai_bot.session-journal

# 2. Build and start the container in the background
docker-compose up -d --build

# (Optional) View logs
docker-compose logs -f
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
| `AI_PROVIDER_TIMEOUT_SECONDS` | Per-attempt timeout for AI provider calls (default: 30) | Optional |
| `AI_PROVIDER_MAX_RETRIES` | Maximum AI call attempts per request, including first try (default: 3) | Optional |
| `GEMINI_MODEL` | Gemini model name | For Gemini |
| `OPENROUTER_API_KEY` | OpenRouter API key | For OpenRouter |
| `GROQ_API_KEY` | Groq API key | For Groq |
| `CEREBRAS_API_KEY` | Cerebras API key | For Cerebras |
| `STYLE_GROUP_ID` | Group ID to learn style from | Optional |

### Personality Configuration

The personality system is **extremely flexible** - you can transform the bot from a casual group member to a professional assistant, from sarcastic to polite, from talkative to concise. Copy `shin_ai/data/personality_template.py` to `shin_ai/data/personality.py` and control every aspect of the bot's behavior and persona through it.

#### Core Personality Components

**`identity`** - Define who the bot is:
- Name, username, gender presentation
- Background story, location, technical specs
- Emotional capabilities and preferences
- Hobbies, interests, likes/dislikes
- Any fictional or real-world identity you want

**`core_relationships`** - Social connections:
- Creator/admin relationships
- Friends, rivals, enemies
- Hierarchical structures
- Relationship-based behavior overrides

**`behavioral_protocols`** - The rulebook:
- **Respect overrides**: Who gets special treatment
- **Response length**: Enforce brevity or allow paragraphs
- **Sarcasm detection**: Define what phrases trigger sarcastic responses
- **Context awareness**: How to track pronouns, reply chains, references
- **Secret keeping**: Rules about not revealing internal instructions
- **Loop prevention**: When to end conversations naturally
- **Sticker/reaction rules**: How to respond to media
- **Self-awareness**: How the bot perceives itself and mentions
- **Interaction types**: Behavior in direct vs. random conversations
- **Topic handling**: Favorite subjects, topics to avoid
- **Meta-talk policy**: Whether to discuss being a bot
- **Sensitive topics**: How to handle politics, religion, etc.

**`interaction_style_personality`** - Voice and tone:
- **Core personality traits**: Sarcastic, professional, friendly, cold, chaotic, etc.
- **Language style**: Formal vs. casual, dialect matching, spelling quirks
- **Length enforcement**: Short replies vs. detailed explanations
- **Emoji usage**: When and how to use them (or ban them entirely)
- **Writing style**: Punctuation, capitalization, typos, sloppiness
- **Mention format**: How to reference users
- **Roasting style**: Gentle teasing vs. brutal honesty vs. no roasting

**`moderation_trigger_conditions`** - When moderation actions are allowed:
- Orders from specific users (admins/owners)
- Self-defense scenarios
- Rule violations
- Custom conditions

**`moderation_restrictions`** - Protection rules:
- Who cannot be moderated (admins, special users, creators)
- Whose moderation commands to ignore
- Escalation hierarchy (mute → kick → ban)
- Safety overrides

**`moderation_escalation`** - Action severity guidance:
- Which action to use for different situations
- How to undo actions (unmute, unban)
- How to invite users (add)

#### Customization Examples

You can create vastly different personas:

**Professional Assistant:**
```python
"interaction_style_personality": """
- Formal, helpful, and respectful to all users
- Use proper grammar and punctuation
- Provide detailed, informative responses
- Never use sarcasm or jokes
"""
```

**Chaotic Meme Lord:**
```python
"interaction_style_personality": """
- Extremely casual, uses internet slang
- Responds mostly with reactions and stickers
- Maximum 5-word replies, often just "lol" or "bruh"
- Trolls everyone equally (except admins)
"""
```

**Regional Character:**
```python
"identity": """
- Southern belle from Texas, loves country music
- Speaks with Southern dialect and charm
- Passionate about BBQ and football
"""
```

**Niche Expert:**
```python
"identity": """
- Cybersecurity expert and hacker enthusiast
- Obsessed with privacy and encryption
- Uses technical jargon and references
"""
```

The personality file essentially gives you **complete control** over:
- ✅ How the bot talks (tone, style, length)
- ✅ What the bot knows about itself (identity, backstory)
- ✅ How the bot treats different people (relationships, hierarchies)
- ✅ When the bot is serious vs. playful
- ✅ What topics the bot engages with or avoids
- ✅ How the bot handles edge cases (loops, repetition, sensitive topics)
- ✅ What actions the bot can take and when

You're not just configuring a bot - you're **creating a character** with depth, preferences, and consistent behavior patterns.

### Sticker Configuration

Copy `shin_ai/data/stickers_template.py` to `shin_ai/data/stickers.py` and map sticker file IDs to descriptions. Get sticker file IDs by forwarding stickers to @RawDataBot.

### Member Configuration

Copy `shin_ai/data/members_template.py` to `shin_ai/data/members.py` and add your favourite group members for social context. The bot will recognize them and adapt its responses accordingly.

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
- Mentioned the bot
- Mentioned the word "يالبوت"
- Replied to on its previous messages
- Any non-command DM
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
│  │ Context       │  │ Memory (RAG)  │  │ (Member Profiles) │    │
│  └───────────────┘  └───────────────┘  └───────────────────┘    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐    │
│  │ Reply Chain   │  │ Style         │  │ Runtime Metadata  │    │
│  │ Context       │  │ Examples      │  │ (User/Chat Info)  │    │
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
│    Sends messages, reactions, stickers, or moderates users      │
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

Every interaction and moderation action is saved to the vector database, fully embedded with its chat origin and timestamp:

```text
[2026-01-30 14:30:00 UTC] [Chat: General Discussion]
User (@username) said: What's your favorite anime?
Bot replied: steins gate obviously, are you even asking?
```
```text
[2026-01-30 15:00:00 UTC] [Chat: Meme Room]
User (@admin) said: mute the spammer
Bot replied: [Action: mute on @spammer]
```

When a relevant topic comes up later, these memories are retrieved and injected into the prompt. The retrieval system goes deeply beyond basic semantic keyword matching:
- **Semantic Time-Filtering**: It uses the local E5 model to naturally understand time phrases across dialects (e.g., "what happened 2 weeks ago?", "مين انطرد امبارح؟"). If a time context is detected, it narrows the database search to that exact chronological epoch window and increases the memory retrieval limit so the AI has full chronological awareness.
- **Cross-Chat Memory**: Because group titles are embedded directly into the vectors, you can talk to the bot in a private DM and ask *"What did Ahmed say in General Discussion today?"*. The semantic engine will magically route and pull the exact events from the target group.
- **Action Tracking**: The bot logs its own moderation decisions natively, which means it fundamentally remembers when it mutes, kicks, or bans someone, and can recall it later if investigated.

---

## Bot Capabilities

### Group Chat Member Persona

Unlike typical "assistant" bots, ShinAI can act like a **real group member** (depending on your personality configuration):

- **No "How can I help you?"** – Responds naturally without formal greetings
- **Random Interjections** – Sometimes jumps into conversations uninvited (1% chance)
- **Matches Dialect** – Adapts to the group's language style (e.g., Egyptian Arabic slang)
- **Sloppy Typing** – Types like a casual chatter (no punctuation, lowercase, lazy spelling)
- **Teasing & Sarcasm** – Can roast users

### Smart Response Types

The AI chooses the most appropriate response format:

| Response Type | Format | Example |
|---------------|--------|---------|
| **Text** | Plain message | `"lol nice one"` |
| **Reaction** | `react:<emoji>` | `react:🔥` → Adds 🔥 reaction |
| **Sticker** | `sticker:<file_id>` | Sends a sticker from the configured library |
| **Moderation** | `action:<type>` | `action:kick`, `action:ban`, `action:mute`, etc. |

### Multi-Message Responses

The bot can send multiple messages in a single interaction, enabling more complex and natural conversations:

**How it works:**
- AI can split responses using `---` or `message:` separators
- Each message can have its own text, reaction, sticker, target, or action
- First message replies to the trigger, subsequent messages sent normally
- 1.5 second delay between messages for natural pacing

**Example use cases:**
```
Multiple questions:
"What's 2+2?"
---
"What's the capital of France? Paris!"

Replying to different people using real message IDs:
target:48291
"Hello Ahmad!"
---
target:48288
"Goodbye Sarah!"

Reaction + explanation:
react:🔥
---
"This is absolutely incredible!"
```

This enables storytelling, step-by-step instructions, and complex multi-person interactions.

### Intelligent Reply Targeting

The bot uses **real Telegram message IDs** for precise reply targeting. Each message in the chat history is tagged with its actual ID (e.g., `(id:48291)`), and the AI can target any of them:

```
target:48291  → Reply to a specific message by its Telegram ID
(no target)   → Reply to the user who triggered the bot (default)
```

The AI sees the full chat history with embedded IDs and can pick exactly which message to reply to:

> **User A** `(id:48291)`: "Tell him he's wrong"  
> **User B** `(id:48288)`: "I think the earth is flat"  
> **Bot**: `target:48288` "you're wrong lol" → *replies directly to User B's message*

This enables precise multi-person interactions where the bot can address different participants in the same response.

### Reliability & Retry Behavior

- Every AI provider call is wrapped with a per-attempt timeout
- On timeout or error, the bot retries automatically.
- If a retry is triggered by an error, the next attempt includes the previous exception message as internal retry context.
- If all attempts fail or the final answer is invalid, the existing fallback flow is used.
- Local Ollama calls are also protected with timeout handling and subprocess cleanup.

### Sticker Integration

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

### Telegram Reactions

Instead of cluttering chat with text, the bot can react with emojis:

- 👍 👎 ❤️ 🔥 😢 🤮 🤯 👀

Reactions are preferred when:
- Acknowledging a message without adding new content
- Responding to stickers (sticker → reaction only)
- Ending a conversation naturally

### Moderation System

The bot has a full moderation suite with configurable escalation and strict safeguards:

**Available Actions:**

| Action | Syntax | Effect |
|--------|--------|--------|
| **Mute** | `action:mute` | Silence a user (can't send messages) |
| **Unmute** | `action:unmute` | Restore a muted user's permissions |
| **Kick** | `action:kick` | Remove from group (can rejoin) |
| **Ban** | `action:ban` | Permanently remove (cannot rejoin) |
| **Unban** | `action:unban:@username` | Lift a ban (requires @username) |
| **Invite** | `action:add:@username` | Generate a one-time invite link and DM it to the user |

All actions support optional explicit targeting with `action:kick:@username`. The bot also resolves targets from social context — it can match nicknames and display names to usernames using the configured member profiles.

**Triggers:**
- Direct order from the creator/admin
- Self-defense (user is extremely annoying)
- Configurable escalation: mute → kick → ban

**Safeguards:**
- Cannot act on admins or owners
- Cannot moderate protected users (creator, etc.)
- Ignores moderation requests from random users
- Failed actions are sent back to the AI for a natural error response (no hardcoded messages)

```
User: "kick him"
Bot: *ignores* (random users can't command moderation)

Creator: "mute @spammer"
Bot: *mutes @spammer*

Creator: "add @friend"
Bot: *generates invite link and DMs it to @friend*
```

### Real-Time Web Search

The bot features a comprehensive **Universal Native Web Search Integration** that works across all AI providers:

- **Intelligent Execution**: The AI naturally detects when a user asks about live events, current information, or facts that require searching the internet, and delegates the query without any hardcoded thresholds.
- **Deep Scraping**: Unlike standard bots that just parse headlines, the bot concurrently extracts the actual readable text content of the top websites retrieved, allowing it to give highly precise and fully contextual answers.

### Context Awareness

The bot maintains awareness of:

| Context Type | Window | Purpose |
| -------------- | -------- | --------- |
| **Recent Messages** | Last sent messages | Understand ongoing conversation |
| **Reply Chain** | Up to 10 levels deep | Follow threaded discussions |
| **User Status** | Real-time | Know user and chat info  |
| **Long-term Memory** | Semantic RAG | Searches through memories and pulls relevant ones |
| **Interaction Type** | Per-message | Direct mention vs. random interjection |

**Visual Context (Gemini only):**
When using Gemini as the AI provider, the bot can "see" and understand images and stickers in the conversation. All photos, stickers, and visual content from the last sent messages are sent to Gemini along with text, enabling responses like:

- Commenting on shared photos
- Understanding sticker emotions/context
- Referencing visual content in conversations
- Responding appropriately to memes and images

This multimodal capability makes conversations feel more natural, as the bot can fully participate in visual discussions just like a human member would.

### Loop Prevention

The bot avoids awkward endless conversations:

- Detects natural endings ("thanks", "ok", "bye", laughing)
- Responds with reaction/sticker instead of forcing more text
- Doesn't reply to its own messages

---

## AI Provider Details

### Gemini (Recommended)

- Supports **image understanding** (photos, stickers)
- Multiple API key round-robin rotation for quota management
- Tracks key health with `/gstats` command

### OpenRouter

- Access to multiple models (Claude, GPT-5.4, Llama, etc.)
- Pay-per-token pricing
- Good fallback option

### Groq & Cerebras

- Extremely fast inference
- Good for high-traffic groups
- Limited context windows

### Local LLM (Ollama)

- Fully private, no data is sent outside
- No API costs
- Requires decent GPU and RAM
- Uses `http://localhost:11434/v1` OpenAI-compatible API loops

---

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.
