# Multi-Message Response Feature

## Overview
The bot can now send **multiple messages** in response to a single user request. This allows for more complex interactions like:
- Sending a series of related messages
- Replying to different people in a conversation thread
- Combining reactions, stickers, and text in creative ways

## How It Works

### 1. **For Bot Users**
Nothing changes! The bot just becomes smarter and can send multiple replies when appropriate.

### 2. **For AI Responses**
The AI can now format responses with multiple messages using separators:

#### Format Option A: Using `---`
```
First message here
---
target:parent
Second message to the parent
---
react:👍
Third message with a reaction
```

#### Format Option B: Using `message:`
```
message:
Hello everyone!
message:
target:parent
This is specifically for you
message:
And this is for the original sender
```

### 3. **Features Per Message**
Each message can independently have:
- **Text content**: Regular message text
- **Reaction**: `react:<emoji>` (e.g., `react:🔥`)
- **Sticker**: `sticker:<file_id>`
- **Target**: `target:<sender|parent|grandparent>`
- **Action**: `action:kick` or `action:kick:@username`

## Examples

### Example 1: Answering Multiple Questions
**User**: "What's 2+2? And what's the capital of France?"

**AI Response**:
```
2+2 = 4
---
The capital of France is Paris! 🇫🇷
```

**Result**: Bot sends two separate messages

---

### Example 2: Replying to Different People
**Context**: User asks bot to "tell Ahmad hello and tell Sarah goodbye"

**AI Response**:
```
target:parent
Hello Ahmad! 👋
---
target:grandparent
Goodbye Sarah! See you later! 👋
```

**Result**: 
- First message replies to Ahmad (parent)
- Second message replies to Sarah (grandparent)

---

### Example 3: Reaction + Text
**User**: Shares something awesome

**AI Response**:
```
react:🔥
---
This is absolutely incredible! I love it!
```

**Result**: 
- Bot reacts with 🔥 to the user's message
- Bot sends a text message

---

### Example 4: Sequential Storytelling
**AI Response**:
```
Let me tell you a story...
---
Once upon a time, there was a brave knight
---
He went on a grand adventure
---
And lived happily ever after! ✨
```

**Result**: Four separate messages sent in sequence

## Technical Implementation

### Modified Files

1. **`shin_ai/core/response_parser.py`**
   - Changed return type from `ParsedResponse` to `list[ParsedResponse]`
   - Added logic to split responses by `---` or `message:` markers
   - Each segment is parsed independently

2. **`shin_ai/core/action_executor.py`**
   - Updated `execute_response()` to handle list of messages
   - Loops through each parsed message and executes it
   - Messages chain together (2nd message replies to 1st if both target sender)

3. **`shin_ai/core/prompt_builder.py`**
   - Added documentation about multi-message format in system prompt
   - AI now knows it can use separators

### Message Chaining Logic
When multiple messages target the same person (default `sender`), they automatically chain:

```
Message 1 → User's message
Message 2 → Message 1
Message 3 → Message 2
```

This creates a natural conversation thread.

## Backward Compatibility

✅ **100% Backward Compatible**

Single-message responses work exactly as before:
- If AI returns plain text, it's treated as one message
- If no separators found, entire response is one message
- Existing prompts don't need any changes

## Testing

Run the test script to verify functionality:
```bash
python test_multi_message.py
```

Expected output shows:
- Single message parsing
- Multiple messages with `---`
- Multiple messages with `message:`
- Mixed content (reactions + text + stickers)

## Use Cases

### When to Use Multiple Messages:
1. **Multiple questions answered separately**
2. **Replying to different people in a thread**
3. **Step-by-step instructions**
4. **Storytelling or jokes with punchlines**
5. **Combining reactions with explanatory text**
6. **List items that deserve separate messages**

### When NOT to Use:
- Single, cohesive thought should stay as one message
- Avoid excessive fragmentation (don't split every sentence)
- Consider readability - too many messages can be spammy

## Configuration

No configuration needed! The feature works automatically.

The AI will decide when to use multiple messages based on:
- Context of the conversation
- User's request structure
- Natural conversation flow

## Troubleshooting

### Issue: Messages not separating
**Solution**: Ensure separators are on their own lines:
```
Good:
Message 1
---
Message 2

Bad:
Message 1 --- Message 2
```

### Issue: Empty messages being sent
**Solution**: The parser automatically filters out empty messages. If you see this, check that each segment has content.

### Issue: Messages sent in wrong order
**Solution**: Messages are sent in the order they appear in the response. Check the separator positions.

## Future Enhancements

Potential improvements:
- [ ] Delay between messages (for dramatic effect)
- [ ] Typing indicators between multi-messages
- [ ] Max message limit per response
- [ ] Message grouping/batching options
- [ ] Per-message media support (images in different messages)

---

**Last Updated**: 2026-02-09
