<div align="center">

# ShinAI

An intelligent multi-platform bot that acts like a real group member - not an assistant. Features personality-driven responses, long-term memory with RAG architecture, style learning, and contextual awareness across Telegram, Discord, and WhatsApp.

[![Chat on Telegram](https://img.shields.io/badge/Chat%20on-Telegram-blue.svg?logo=telegram)](https://t.me/shinobi7kbot)
[![Add to Discord](https://img.shields.io/badge/Add%20to-Discord-5865F2?logo=discord)](https://discordapp.com/users/855437723166703616)

</div>

## Table of Contents
- [Inspiration](#inspiration)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Multi-Platform Support](#multi-platform-support)
- [Triggers](#triggers)
- [Bot Capabilities](#bot-capabilities)
  - [Group Chat Member Persona](#group-chat-member-persona)
  - [Smart Response Types](#smart-response-types)
  - [Multi-Message Responses](#multi-message-responses)
  - [Intelligent Reply Targeting](#intelligent-reply-targeting)
  - [Reliability & Retry Behavior](#reliability--retry-behavior)
  - [Reactions & Stickers](#reactions--stickers)
  - [Moderation System](#moderation-system)
  - [Real-Time Web Search](#real-time-web-search)
  - [Context Awareness](#context-awareness)
- [How It Works](#how-it-works)
  - [Architecture Overview](#architecture-overview)
  - [Vector Embeddings & ChromaDB](#vector-embeddings--chromadb)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Memory System](#memory-system)
  - [Real-Time Web Search](#real-time-web-search-1)
  - [Context Awareness](#context-awareness-1)
  - [Loop Prevention](#loop-prevention)
- [AI Provider Details](#ai-provider-details)
- [License](#license)


## Inspiration

I was tired of AI bots that felt like... bots. You know, the type: "As an AI language model, I cannot..." or the overly formal "How can I assist you today?" that kills the vibe in a casual group chat with friends.

I wanted an AI that didn't just stand on the sidelines waiting for a command, but actually **lived** in the group chat. I wanted a "digital homie". Someone who knows the inside jokes, understands the group's specific slang, remembers that embarrassing thing you said three weeks ago, and isn't afraid to roast you for it. And definitely without being so cringe and unbearable. **ShinAI** was born from the desire to bridge the gap between "helpful assistant" and "chaotic group member."

## Features

- 🧠 **Multiple AI Providers**: Gemini, OpenRouter, Groq, Cerebras, or local LLM (Ollama)
- 🧩 **Multi-Platform Support**: Seamlessly functions on **Telegram**, **Discord**, and **WhatsApp** simultaneously.
- 💬 **Personality System**: Fully customizable bot personality and behavior
- 🎭 **Social Context**: Recognizes group members across platforms and adapts responses
- 🔄 **Reply Chain Tracking**: Understands conversation context with deep history retrieval
- 📝 **Long-term Memory**: Remembers past conversations using vector embeddings (cross-platform awareness)
- 🎨 **Style Learning**: Learns communication patterns from example messages
- 🌐 **Real-Time Web Search**: Web searching, fetching, and scraping capabilities used when needed
- 📌 **Sticker Support**: Send stickers as responses with custom mappings
- 😀 **Emoji Reactions**: React to messages with emojis instead of text
- 📨 **Multi-Message Responses**: Send multiple sequential messages with automatic delays
- 🎯 **Smart Reply Targeting**: Target any message in the chat by its real platform ID
- 🛡️ **Moderation Suite**: Kick, ban, unban, mute, unmute, and invite users with platform-native actions
- 🔁 **Reliability Layer**: Per-attempt timeout, retries mechanism, and error-aware retry context injection
- ⚡ **Rate Limiting**: Built-in cooldowns to prevent spam

## Quick Start

You can either chat with the bot on [Telegram](https://t.me/shinobi7kbot) or [Discord](https://discordapp.com/users/855437723166703616) using its [triggers](#triggers) and add it to your groups/servers, or build it from source yourself:

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
# 1. Add STYLE_GROUP_ID to your .env (get group ID from @RawDataBot or Discord Developer Portal)
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
# 1. First time setup: create empty session files for Telegram
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
| `TELEGRAM_ENABLED` | Enable or disable Telegram platform (`true`/`false`) | Optional |
| `DISCORD_ENABLED` | Enable or disable Discord platform (`true`/`false`) | Optional |
| `WHATSAPP_ENABLED` | Enable or disable WhatsApp platform (`true`/`false`) | Optional |
| `TELEGRAM_API_ID` | Telegram API ID from my.telegram.org | For Telegram |
| `TELEGRAM_API_HASH` | Telegram API Hash | For Telegram |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot token from @BotFather | For Telegram |
| `DISCORD_BOT_TOKEN` | Discord Bot Token from Developer Portal | For Discord |
| `ADMIN_USER_ID` | Your principal user ID (Telegram or Discord) | ✅ |
| `AI_CHOICE` | AI provider (gemini/openrouter/groq/cerebras/local/manual) | ✅ |
| `AI_PROVIDER_TIMEOUT_SECONDS` | Per-attempt timeout for AI provider calls (default: 60) | Optional |
| `AI_PROVIDER_MAX_RETRIES` | Maximum AI call attempts per request (default: 3) | Optional |
| `MIN_REPLY_DELAY_SECONDS` | Minimum random delay before responding (seconds) | Optional |
| `MAX_REPLY_DELAY_SECONDS` | Maximum random delay before responding (seconds) | Optional |
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
 
 Copy `shin_ai/data/members_template.py` to `shin_ai/data/members.py` and add your favourite group members for social context. The bot will recognize them across Telegram, Discord and WhatsApp and adapt its responses accordingly.
 
 #### Platform Mapping
 You can define platform-specific handles for each member:
 - `telegram_username`: The user's Telegram handle (without @).
 - `discord_username`: The user's Discord username (not display name).
 
 This allows the bot to remember someone on Telegram even if they are using a different username on Discord.

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

## Multi-Platform Support
 
 ShinAI is architected as a **Multi-Platform AI Agent**. It uses a specialized Platform Adapter layer that decouples the AI logic from the specific platform SDK.
 
 | Feature | Telegram Capability | Discord Capability |
 |---------|--------------------|--------------------|
 | **Reactions** | Native Emoji Reactions | Native Emoji Reactions |
 | **Stickers** | Full Support (mapped) | Not supported (Text fallback) |
 | **Media** | Photo/Video Awareness | Attachment Awareness (Text context) |
 | **Moderation** | Ban, Kick, Mute, Invite | Native Ban, Kick, Timeout |
 | **Identity** | Unified via Telegram ID | Unified via Discord ID |
 
 The most powerful feature of this architecture is **Unified Memory**. If you talk to the bot on Discord about a conversation that happened on Telegram, the bot will retrieve those memories and answer correctly, understanding exactly which interaction occurred on which platform.

## Triggers

The bot responds when:
- Mentioned the bot
- Mentioned the word "يالبوت"
- Replied to on its previous messages
- Any non-command DM
- Random chance on any group message

 ---
 
 ## Bot Capabilities
 
 ### Group Chat Member Persona
 
 Unlike typical "assistant" bots, ShinAI acts like a **real group member**:
 
 - **No "How can I help you?"** – Responds naturally without formal greetings.
 - **Random Interjections** – Sometimes jumps into conversations uninvited.
 - **Matches Dialect** – Adapts to the group's language style.
 - **Sloppy Typing** – Types like a casual chatter (no punctuation, lowercase, lazy spelling).
 - **Teasing & Sarcasm** – Can roast users effectively.
 
 ### Smart Response Types
 
 The AI chooses the most appropriate response format for the platform:
 
 | Response Type | Format | Platform Notes |
 |---------------|--------|----------------|
 | **Text** | Plain message | Supported everywhere |
 | **Reaction** | `react:<emoji>` | Supported on Telegram & Discord |
| **Sticker** | `sticker:<id>` | Telegram supports native sticker IDs; WhatsApp supports `sticker:wa:<url_or_local_path>`; ignored on Discord |
 | **Moderation** | `action:<type>` | Translated to native actions (e.g. Timeout on Discord) |
 
 ### Multi-Message Responses
 
 The bot can send multiple messages in a single interaction for more natural pacing:
 - AI can split responses using `---` or `message:` separators.
 - A few seconds delay between messages.
 - Each message can target different users or perform different actions.
 
 ### Intelligent Reply Targeting
 
 The bot uses **Unified Message IDs** for precise reply targeting. The AI sees the chat history with embedded IDs and can pick exactly which message to reply to:
 
 ```
 target:12345  → Reply to a specific message ID from the platform
 (no target)   → Reply to the user who triggered the bot (default)
 ```

 ### Reliability & Retry Behavior
 
 - Every AI provider call is protected with a per-attempt timeout and automatic retries.
 - If an attempt fails, the previous error context is injected into the next attempt so the AI can adapt.
 - If all attempts fail, a graceful fallback or manual mode is triggered.
 
### Reactions & Stickers

- **Reactions**: Preferred for acknowledging messages or ending conversations without text.
- **Stickers (Telegram)**: Selected based on emotional context from a custom library.
- **Stickers (WhatsApp)**: Use `sticker:wa:<https-url-or-local-path>` so the adapter can send media through Neonize `send_sticker`.
- **Discord**: Sticker actions are ignored and naturally degrade to text/reaction behavior.
 
 ### Moderation System
 
 The bot features a full moderation suite with platform-native actions:
 
 | Action | Telegram Effect | Discord Effect | WhatsApp Effect |
 |--------|-----------------|----------------|
 | **Mute** | Restricted Permissions | Native Timeout | No Native Solution |
 | **Kick** | Remove from Group | Native Kick | Native Kick |
 | **Ban** | Permanent Ban | Native Ban | No Native Solution |
 | **Unban** | Lift Ban | Native Unban | No Native Solution |
 | **Invite** | Invite Link | DM Invite Link | No Native Solution |
 
 **Safeguards**:
 - Cannot act on admins or owners.
 - Ignores moderation commands from unauthorized users.
 - Failed actions result in a natural AI-driven explanation.
 
 ### Real-Time Web Search
 
 All AI providers can trigger web searches when real-time information is needed. The bot fetches and scrapes top search results to provide grounded, factual responses.
 
 ### Context Awareness
 
 | Context Type | Purpose |
 | -------------- | --------- |
 | **Recent Context** | Recent messages in the chat for immediate flow |
 | **Reply Chain** | Deeply follows threaded discussions (up to 10 levels) |
 | **Long-term Memory** | Semantic retrieval from the entire history database |
 | **Visual Context** | (Gemini only) Sees photos and stickers on Telegram |

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

### Vector Embeddings & ChromaDB
 
 ShinAI uses **vector embeddings** to enable semantic search across memories and member profiles. Here's how it works:
 
 1. **Text → Vector**: Text is converted into high-dimensional vectors (embeddings) using `sentence-transformers` with the `intfloat/multilingual-e5-large` model.
 2. **Semantic Similarity**: Similar concepts cluster together in vector space, enabling "meaning-based" search rather than keyword matching.
 3. **ChromaDB Storage**: Vectors are stored in ChromaDB, a lightweight vector database optimized for embedding search.
 
 ```python
 # Example: "Who created you?" matches the creator's profile
 # even without exact keyword matches like "creator"
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
 
 Every interaction and moderation action is saved to the vector database, tagged with its platform and chat origin:
 
 ```text
 [2026-01-30 14:30:00 UTC] [Platform: Discord] [Chat: General]
 User (@username) said: What's your favorite anime?
 Bot replied: steins gate obviously, are you even asking?
 ```
 
 When a relevant topic comes up later, these memories are retrieved and injected into the prompt. The retrieval system is platform-aware but not platform-isolated:
 - **Cross-Platform Retrieval**: You can ask on Telegram *"What did Sarah say on Discord yesterday?"* and the bot will pull the correct fragments from the Discord history.
 - **Semantic Time-Filtering**: It understands time phrases across dialects. If a time context is detected, it narrows the database search to that exact chronological window.
 - **Action Tracking**: The bot logs its own moderation decisions natively, remembering when it muted or banned someone across all platforms.

### Real-Time Web Search

The bot features a comprehensive **Native Web Search Integration** that works across all AI providers using the `duckduckgo-search` library paired with `beautifulsoup4` and `httpx`.

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
