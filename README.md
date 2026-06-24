<div align="center">

# ShinAI

An intelligent multi-platform bot that acts like a real group member - not an assistant. Features personality-driven responses, long-term memory with RAG architecture, style learning, and contextual awareness across Telegram, Discord, and WhatsApp.

[![Telegram](https://img.shields.io/badge/Telegram-Chat-26A5E4?style=for-the-badge&logo=telegram&logoColor=white&labelColor=1f2937)](https://t.me/shinobi7kbot)
[![Discord](https://img.shields.io/badge/Discord-Add%20Me-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=1f2937)](https://discordapp.com/users/855437723166703616)

[![Interactions](https://img.shields.io/badge/50K%2B-Interactions-FF6B35?style=for-the-badge&logo=sparkfun&logoColor=white&labelColor=1f2937)]()
[![Ko-Fi](https://img.shields.io/badge/Ko--fi-Support%20Me-FF5E5B?style=for-the-badge&logo=kofi&logoColor=white&labelColor=1f2937)](https://ko-fi.com/MAymanKH)

</div>

## Table of Contents
- [Inspiration](#inspiration)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Multi-Platform Support](#multi-platform-support)
- [Triggers](#triggers)
- [Capabilities & Technical Deep Dive](#capabilities--technical-deep-dive)
  - [Architecture Overview](#architecture-overview)
  - [Group Chat Member Persona](#group-chat-member-persona)
  - [Smart Response Types & Platform Tool-Calling](#smart-response-types--platform-tool-calling)
  - [Memory & Retrieval System](#memory--retrieval-system)
  - [Context Awareness & RAG Engine](#context-awareness--rag-engine)
  - [Real-Time Web Search](#real-time-web-search)
  - [Conversational Dynamics](#conversational-dynamics)
  - [Audio Processing Pipeline](#audio-processing-pipeline)
  - [Reliability & Retries](#reliability--retries)
- [AI Provider Details](#ai-provider-details)
- [License](#license)


## Inspiration

I was tired of AI bots that felt like... bots. You know, the type: "As an AI language model, I cannot..." or the overly formal "How can I assist you today?" that kills the vibe in a casual group chat with friends.

I wanted an AI that didn't just stand on the sidelines waiting for a command, but actually **lived** in the group chat. I wanted a "digital homie". Someone who knows the inside jokes, understands the group's specific slang, remembers that embarrassing thing you said three weeks ago, and isn't afraid to roast you for it. And definitely without being so cringe and unbearable. **ShinAI** was born from the desire to bridge the gap between "helpful assistant" and "chaotic group member."

## Features

- 🧠 **Multiple AI Providers**: Gemini, OpenRouter, Groq, Cerebras, Ollama, or any OpenAI-compatible API
- 🧩 **Multi-Platform Support**: Seamlessly functions on **Telegram**, **Discord**, and **WhatsApp** simultaneously.
- 💬 **Personality System**: Fully customizable bot personality and behavior
- 🎭 **Social Context**: Recognizes group members across platforms and adapts responses
- 🔄 **Reply Chain Tracking**: Understands conversation context with deep history retrieval
- 📝 **Long-term Memory**: Remembers past conversations using vector embeddings
- 🎨 **Style Learning**: Learns communication patterns from example messages
- 🌐 **Real-Time Web Search**: Web searching, fetching, and scraping capabilities used when needed
- 🎙️ **Voice Message Transcription**: Local, fast, and free audio transcription using faster-whisper
- 📌 **Sticker Support**: Send stickers as responses with custom mappings
- 😀 **Emoji Reactions**: React to messages with emojis instead of text
- 📨 **Multi-Message Responses**: Send multiple sequential messages with automatic delays
- 🎯 **Smart Reply Targeting**: Target any message in the chat by its real platform ID
- 🤔 **Speculative Responses**: Intelligently jumps into ongoing conversations when appropriate
- ⏱️ **Human-like Delays**: Randomizes response times to simulate human reading and typing
- 🛡️ **Moderation Suite**: Kick, ban, unban, mute, unmute, and invite users with platform-native actions
- 🔁 **Reliability Layer**: Per-attempt timeout, retries mechanism, and error-aware retry context injection
- ⚡ **Rate Limiting**: Built-in cooldowns to prevent spam

## Quick Start

You can either chat with the bot on [Telegram](https://t.me/shinobi7kbot) or [Discord](https://discordapp.com/users/855437723166703616) using its [triggers](#triggers) and add it to your groups/servers, or build it from source yourself:

### 0. Install Prerequisites

- [Python](https://www.python.org/downloads/) 3.10 or higher
- [Git](https://git-scm.com/install/)
- Make sure they are added to your path.

### 1. Clone the Repository

```bash
git clone https://github.com/MAymanKH/ShinAI.git
cd ShinAI
```

### 2. Create a Virtual Environment (Recommended)

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
cp config.yaml.example config.yaml
# Edit config.yaml with your credentials and settings
```

### 5. Customize Your Bot

Copy the template files and customize them:

```bash
# Personality configuration
cp shin_ai/data/personality_template.py shin_ai/data/personality.py

# Sticker mappings (optional)
cp shin_ai/data/stickers_template.py shin_ai/data/stickers.py
```

### 6. Index Style Examples (Optional)

If you want the bot to learn from a specific group's communication style:

```bash
# 1. Add style_group_id to your config.yaml (get Telegram's group ID from @RawDataBot)
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

ShinAI is configured using a centralized `config.yaml` file. Copy `config.yaml.example` to `config.yaml` and fill in your details.

### Configuration Structure

The structure of the `config.yaml` configuration is shown below as a commented template:

```yaml
# Platform Settings
platform:
  telegram:
    enabled: true           # Enable/disable Telegram client (true/false)
    api_id: 123456          # API ID from https://my.telegram.org
    api_hash: "your_hash"   # API Hash from https://my.telegram.org
    bot_token: "your_token" # Bot token from @BotFather
  discord:
    enabled: false          # Enable/disable Discord client (true/false)
    bot_token: "your_token" # Discord Bot token from the Developer Portal
  whatsapp:
    enabled: false          # Enable/disable WhatsApp (will link session via console QR code)

# Administration User ID
admin_user_id: 123456789    # Platform user ID (Telegram or Discord) allowed to run admin commands

# Verbose Logging Toggle
debug: false                # Set to true to enable detailed log outputs

# Response Timings & Probability
response:
  min_delay_seconds: 5.0    # Lower boundary for randomized response delay (simulate typing/reading)
  max_delay_seconds: 300.0  # Upper boundary for randomized response delay
  random_trigger_probability: 0.05 # Random chance (0.0 to 1.0) to respond to normal group messages without mention

# Voice Transcription Service (Whisper)
whisper:
  model: large-v3-turbo     # Model size to load: tiny, base, small, medium, large-v2, large-v3-turbo
  language: auto            # Language code (e.g., 'ar', 'en') or 'auto' to auto-detect
  cpu_threads: 2            # Number of CPU threads dedicated to Whisper inference

# Web Search Settings
# firecrawl:
#   api_key: "fc-your-key-here"                   # (Optional - fallback to duckduckgo if not configured)

# Semantic Retrieval & Style Settings
embedding_model: intfloat/multilingual-e5-large # Transformer model used for memories and style indexing
style_group_id: -1001234567890                  # (Optional) Telegram group ID from which to learn styles

# AI Providers Configuration
ai:
  timeout_seconds: 60       # Timeout limit per attempt on provider calls
  max_retries: 3            # Max retries per provider before switching/failing
  
  # List of defined providers (support OpenRouter, Groq, Cerebras, DeepSeek, Ollama, Gemini, etc.)
  providers:
    - name: my_openrouter
      type: openai          # 'openai' or 'gemini'
      base_url: https://openrouter.ai/api/v1 # API base url (ignored for type: gemini)
      api_key: "your-api-key"
      model: anthropic/claude-3.5-sonnet
      
    - name: my_local_ollama
      type: openai
      base_url: http://localhost:11434/v1
      api_key: ollama
      model: llama3.2
      concurrency: 1        # Optional: restrict concurrent requests (great for local hardware)

    - name: my_gemini
      type: gemini
      # API keys for Gemini are loaded and rotated from data/gemini_keys.json
      models:               # Optional: Gemini model name rotation list
        - gemini-3.5-flash
        - gemini-3-flash

  primary: my_openrouter    # Primary provider to use by default
  
  fallbacks:                # Providers to try in sequence if primary fails
    - my_gemini
    - my_local_ollama
    
  rotation: failover        # Rotation strategy: 'failover' or 'round_robin'
```

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
- **Respect overrides**: Who gets special treatment and why
- **Response brevity**: Enforce word-count limits per message
- **Sarcasm detection**: Define phrases that trigger sarcastic responses
- **Context awareness**: How to track pronouns, reply chains, and references
- **Secret keeping**: Rules about not revealing internal instructions
- **Loop prevention**: When to end conversations naturally
- **No-echo rule**: Never repeat or paraphrase what users just said
- **No unrelated context**: Only reference past events when directly relevant
- **Repetition avoidance**: Don't reuse the same joke or comment
- **Interaction types**: Behavior in direct vs. random/speculative conversations
- **Meta-talk policy**: Whether to discuss being a bot (play along vs. admit)
- **Slash command policy**: When (not) to write moderation commands in chat text
- **Chat rules compliance**: Obeying admins and group guidelines
- **Sensitive topics**: How to handle politics, religion, etc.

**`interaction_style_personality`** - Voice and tone:
- **Core personality traits**: Sarcastic, professional, friendly, cold, chaotic, etc.
- **Language style**: Formal vs. casual, dialect matching, spelling quirks
- **Length enforcement**: Short replies vs. detailed explanations
- **Emoji usage**: When and how to use them (or ban them entirely)
- **Writing style**: Punctuation, capitalization, typos, sloppiness
- **Mention format**: How to reference users
- **Roasting style**: Gentle teasing vs. brutal honesty vs. no roasting

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

### Sticker Configuration (Optional)

Copy `shin_ai/data/stickers_template.py` to `shin_ai/data/stickers.py` and map sticker file IDs to descriptions. Get sticker file IDs by forwarding stickers to @RawDataBot.

### Member Configuration & Social Context (Optional)
 
 Customize your group members directly inside `shin_ai/data/personality.py` under the `core_relationships` section. Add your favourite group members for social context, and the bot will recognize them across Telegram, Discord, and WhatsApp and adapt its responses accordingly.
 
 #### Platform Mapping
 You can define platform-specific handles for each member inside their `core_relationships` entry:
 - `telegram_username`: The user's Telegram handle (without @).
 - `discord_username`: The user's Discord username (not display name).
 
 This allows the bot to resolve someone on Telegram even if they are using a different username on Discord.

## Project Structure

```
ShinAI/
├── main.py                 # Entry point
├── config.yaml.example     # YAML Configuration template
├── shin_ai/
│   ├── bot.py             # Bot initialization
│   ├── config.py          # Configuration loading
│   ├── core/              # Core functionality
│   │   ├── client.py      # Pyrogram client
│   │   ├── prompt_builder.py
│   │   ├── action_executor.py
│   │   └── state.py
│   ├── data/              # Data templates
│   │   ├── personality_template.py
│   │   ├── stickers_template.py
│   │   └── loader.py
│   ├── handlers/          # Message handlers
│   │   ├── chat.py
│   │   ├── fun.py
│   │   └── stats.py
│   ├── providers/         # AI providers
│   │   ├── gemini.py
│   │   ├── gemini_keys.py
│   │   ├── manual.py
│   │   ├── openai_compatible.py
│   │   ├── registry.py
│   │   └── tool_loop.py
│   ├── services/          # Business logic
│   │   ├── social.py
│   │   └── replies.py
│   ├── stylers/           # Style learning
│   │   ├── style_indexer.py
│   │   └── style_retriever.py
│   └── utils/             # Utilities
│       ├── action_tools.py
│       ├── context_manager.py
│       ├── db.py
│       ├── logger_config.py
│       ├── memory.py
│       ├── memory_lookup.py
│       ├── memory_lookup_filters.py
│       ├── memory_time.py
│       ├── rate_limit.py
│       └── web_search.py
└── data/                  # Runtime data (gitignored)
    ├── gemini_keys.json
    ├── gemini_stats.json
    └── bot_replies.json
```

## Multi-Platform Support
 
 ShinAI is architected as a **Multi-Platform AI Agent**. It uses a specialized Platform Adapter layer that decouples the AI logic from the specific platform SDK.
 
 | Feature | Telegram Capability | Discord Capability | WhatsApp Capability |
 |---------|--------------------|--------------------|---------------------|
 | **Reactions** | Supported | Supported | Supported |
 | **Stickers** | Supported | Not supported | Supported |
 | **Media** | Supported | Supported | Supported |
 | **Moderation** | Ban, Kick, Mute, Invite | Ban, Kick, Timeout | Kick |
 | **Identity** | Unified via Telegram ID | Unified via Discord ID | Unified via WhatsApp JID |
 
 The most powerful feature of this architecture is **Unified Memory**. If you talk to the bot on Discord about a conversation that happened on Telegram, the bot will retrieve those memories and answer correctly, understanding exactly which interaction occurred on which platform.

## Triggers

The bot responds when:
- Mentioned the bot
- Mentioned the word "يالبوت"
- Replied to on its previous messages
- Any non-command DM
- Speculative interaction when following up on its own messages
- Random chance on any group message

 ---
 
  ## Capabilities & Technical Deep Dive

ShinAI combines conversational intelligence, platform integrations, RAG (Retrieval-Augmented Generation), and real-time execution tools into a unified agent. Below is a detailed view of its core capabilities and inner architecture.

### Architecture Overview

ShinAI uses a **Retrieval-Augmented Generation (RAG)** architecture to create contextually-aware responses. Rather than relying solely on the AI model's training data, the bot retrieves relevant information from multiple sources before generating a response.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Incoming Message                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                      Context Collection                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐ │
│  │ Recent Chat   │  │ Long-term     │  │ Social Context    │ │
│  │ Context       │  │ Memory (RAG)  │  │ (Member Profiles) │ │
│  └───────────────┘  └───────────────┘  └───────────────────┘ │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐ │
│  │ Reply Chain   │  │ Style Example │  │ Runtime Metadata  │ │
│  │ Context       │  │   (Optional)  │  │ (User/Chat Info)  │ │
│  └───────────────┘  └───────────────┘  └───────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    System Prompt Builder                        │
│         Combines personality + context + instructions           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AI Provider                            │
│        Runs tool-calling generation loop with tools:            │
│           web_search, memory_lookup, send_reaction              │
│                send_sticker, moderate_user                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Action Executor                             │
│     Executes text responses and deferred pending actions        │
│          (reactions, stickers, or moderation actions)           │
└─────────────────────────────────────────────────────────────────┘
```

### Group Chat Member Persona

Unlike typical formal assistant bots, ShinAI behaves like a natural participant in the chat:
- **No Formalities**: It avoids standard AI boilerplate (e.g., "How can I help you?") and responds naturally.
- **Random Interjections**: Can jump into ongoing conversations without being explicitly mentioned (speculative responses).
- **Style Alignment**: Adapts to the group's specific language style and dialect.
- **Sloppy Typing**: Uses lowercase, lazy spelling, and minimal punctuation to simulate a phone or casual chat user.
- **Personality System**: Supports sarcasm, humor, and teasing, customizable in `shin_ai/data/personality.py`.

### Smart Response Types & Platform Tool-Calling

The AI interacts with chat platforms natively using formal function-calling tools rather than raw text parsing:

| Tool | Parameters | Purpose | Platform Notes |
|---------------|------------|---------|----------------|
| `send_reaction` | `emoji`, `message_id` | Reacts to a message with an emoji | Telegram ✅ WhatsApp ✅ Discord ✅ |
| `send_sticker` | `sticker_id`, `reply_to_message_id` | Sends a sticker from the custom library | Telegram ✅ WhatsApp ✅ Discord ❌ |
| `moderate_user` | `action`, `target_username`, `target_message_id` | Restricts or manages group members | kick (all ✅), ban/unban/mute/unmute (Telegram & Discord ✅), add (Telegram ✅) |

#### Stickers & Reactions
- **Reactions**: Preferred for closing conversation loops, acknowledging messages, or ending a laugh chain.
- **Stickers**: Chosen based on emotional context.

#### Native Moderation System
The bot features native moderation capabilities mapped across platforms:

| Action | Telegram Effect | Discord Effect | WhatsApp Effect |
|--------|-----------------|----------------|-----------------|
| **Mute** | Restricted Permissions | Native Timeout | No Native Solution |
| **Kick** | Remove from Group | Native Kick | Native Kick |
| **Ban** | Permanent Ban | Native Ban | No Native Solution |
| **Unban** | Lift Ban | Native Unban | No Native Solution |
| **Invite** | Invite Link | DM Invite Link | No Native Solution |

- **Safeguards**: The bot cannot act on admins/owners, ignores command triggers from unauthorized users, and falls back to a natural AI response explaining any execution failures.

### Memory & Retrieval System

ShinAI saves every interaction and moderation action to a local vector database:

```text
[2026-01-30 14:30:00 UTC] [Platform: Discord] [Chat: General]
User (@username) said: What's your favorite anime?
Bot replied: steins gate obviously, are you even asking?
```

- **Vector Embeddings & ChromaDB**: Message and style text are converted into dense vector representations using `sentence-transformers` (defaulting to the multilingual `intfloat/multilingual-e5-large` model) and indexed inside a local ChromaDB instance.
- **Cross-Platform Recall**: Retrieve memories across platforms (e.g., asking on Telegram *"What did you say on Discord yesterday?"*).
- **Semantic Time Filtering**: The retrieval pipeline for the long-term memory retrieval (not the lookup tool call) parses relative time phrases (e.g., "yesterday", "three weeks ago") across dialects to isolate searches to exact chronological windows.
- **Memory Lookup Tool**: A dedicated tool exposed directly to the AI for advanced retrieval. It accepts filters like `keywords`, `usernames`, `chat_titles`, `platform`, `time_start`, and `time_end`. Results are re-ranked semantically and filtered using a Maximal Marginal Relevance (MMR) selection pass to ensure diversity.

### Context Awareness & RAG Engine

The bot maintains a rich, multi-layered context window to understand the chat state before responding:

| Context Type | Window / Scope | Purpose |
|--------------|----------------|---------|
| **Recent Context** | Last chat messages | Standard chat history to maintain immediate flow. |
| **Reply Chain** | Up to 10 levels deep | Deeply follows nested threaded discussions. |
| **User Status** | Real-time | Detects usernames, mentions, roles, and platform permissions. |
| **Long-term Memory** | Semantic database | Automatically retrieves past chat fragments related to the query. |
| **Style Examples** | Semantic database | Injects matching past chat messages to guide response style. |
| **Interaction Type** | Per-message | Distinguishes between direct mentions, replies, or speculative interjections. |

#### Visual Context (Multimodal)
- **Native Gemini Vision**: If using Gemini, the bot directly processes incoming photos and stickers. It can comment on images, understand sticker emotions, and reference visual memes.
- **OpenAI-Compatible Vision & Fallback**: OpenAI-compatible providers that support vision receive image assets as base64-encoded data URIs. For text-only providers (e.g., Groq, Cerebras), ShinAI automatically utilizes the primary Gemini provider to generate a textual description of the media asset, appending it to the prompt. Text-only models can also invoke the `ask_gemini_about_image` tool to ask questions about visual attachments dynamically.

### Real-Time Web Search

When the bot identifies a query requiring live data (e.g., current news, weather, or real-time lookup), the AI calls the web search tool:
- **Optional Firecrawl Integration**: If configured in `config.yaml`, the bot first attempts to query the Firecrawl Search API. This automatically fetches clean markdown-formatted pages for the top search results in a single, efficient API call.
- **DuckDuckGo & Custom Scraping Fallback**: If Firecrawl is not configured, or if the API call fails (e.g., quota limits exhausted, invalid API key, network issues, or timeouts), the bot gracefully falls back to using the `duckduckgo-search` library to pull top search results and concurrently scrapes the page contents using `httpx` and `beautifulsoup4`.

### Conversational Dynamics

#### Speculative Responses (Pre-flight Evaluation)
To avoid invading group chats inappropriately, any message that might be a continuation of a conversation with the bot (but doesn't explicitly tag it) undergoes a quick, cost-effective "pre-flight" speculative check. A boolean classifier evaluates the recent context and user intent, determining if the bot should jump in.

#### Queuing & Dynamic Delays
- **Response Delay**: To simulate typing and reading, replies are held for a randomized delay configured via `config.yaml` (`min_delay_seconds` and `max_delay_seconds` under `response`). Set these to `0.0` to disable the delay entirely. Messages received during this window are queued and processed in order.
- **Inter-Message Typing Delay**: When sending multiple messages split by `---`, the bot calculates typing speeds based on text length (~8-9 characters per second) with Gaussian jitter, triggering the platform's native "typing" indicators.

### Audio Processing Pipeline

When a voice message or audio file is received, the platform adapter routes the raw audio bytes to the built-in transcription service. It runs `faster-whisper` inference on a dedicated thread pool to prevent blocking the async event loop. Once the audio is transcribed into text, it is injected into the standard message processing pipeline as if the user had typed it, maintaining a seamless conversational flow.

### Reliability & Retries

To ensure uninterrupted uptime:
- All AI provider completions are wrapped in a configurable per-attempt timeout and retry policy.
- If a provider request fails, the exception details are injected into the retry context, allowing the AI layer to adapt to previous errors.
- If the primary provider fails completely, the registry executes `failover` or `round_robin` strategies against configured fallback APIs.

---

## AI Provider Details

ShinAI supports two main types of AI providers under the hood: Google Gemini (native SDK) and any OpenAI-Compatible API.

### Gemini
- Supports native **multimodal image understanding** (photos, stickers).
- Supports API key rotation (manage keys in `data/gemini_keys.json` to rotate and track quota/health via `/gstats`).
- Configured using the `gemini` provider type in `config.yaml`.

### OpenAI-Compatible APIs
Any provider offering an OpenAI-compliant completions endpoint is supported under the `openai` provider type. This unifies external API services and local inference engines:
- **Commercial API Hubs**: OpenRouter (Claude, GPT, Llama), DeepSeek, Together AI, Fireworks, Groq, Cerebras.
- **Local Inference**: Ollama, vLLM, LM Studio, `text-generation-webui`.
- **Multimodal Visual Fallback**: For providers that do not natively support vision (like Groq/Cerebras), the bot automatically calls the configured Gemini provider to generate a detailed text description of any attached images.
- **Multimodal Image Tool**: Text-only models can dynamically call the `ask_gemini_about_image` tool during conversation to ask specific questions about the visual context.

---

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.
