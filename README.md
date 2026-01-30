# ShinAI 🤖

A customizable Telegram bot powered by multiple AI providers with personality, memory, and social context awareness.

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
git clone https://github.com/yourusername/ShinAI.git
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
