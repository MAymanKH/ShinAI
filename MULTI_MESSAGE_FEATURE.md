# Multi-Message Response Feature

## Overview
The bot can now send **multiple messages** in response to a single user request. This allows for more complex interactions like:
- Sending a series of related messages
- Replying to different people in a conversation thread
- Combining reactions, stickers, and text in creative ways

## How It Works

### 1. **For Bot Users**
Nothing changes! The bot just becomes smarter and can send multiple replies when appropriate.

**Note**: When sending multiple messages:
- Only the **first message** will be a reply to your message
- Subsequent messages will be sent normally to the chat (not as replies)
- There's a **1.5 second delay** between each message for natural pacing

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

**Result**: 
- **Message 1** (0.0s): "Let me tell you a story..." ← replies to user
- **Delay** (1.5s) 
- **Message 2** (1.5s): "Once upon a time..." ← sent normally
- **Delay** (1.5s)
- **Message 3** (3.0s): "He went on a grand adventure" ← sent normally  
- **Delay** (1.5s)
- **Message 4** (4.5s): "And lived happily ever after! ✨" ← sent normally

Total time: ~4.5 seconds for natural story delivery

## Technical Implementation

### Modified Files

1. **`shin_ai/core/response_parser.py`**
   - Changed return type from `ParsedResponse` to `list[ParsedResponse]`
   - Added logic to split responses by `---` or `message:` markers
   - Each segment is parsed independently

2. **`shin_ai/core/action_executor.py`**
   - Updated `execute_response()` to handle list of messages
   - Loops through each parsed message and executes it
   - **Only first message is a reply** - subsequent messages sent normally
   - **1.5 second delay** between messages for natural conversation flow
   - Added `asyncio.sleep()` between message sends
   - Modified `_execute_text()` and `_execute_sticker()` to handle optional `reply_to_id`

3. **`shin_ai/core/prompt_builder.py`**
   - Added documentation about multi-message format in system prompt
   - AI now knows it can use separators

### Message Chaining Logic
When sending multiple messages:

```
User's message
    ↓ (reply)
Message 1
    ↓ (1.5s delay, no reply)
Message 2
    ↓ (1.5s delay, no reply)
Message 3
```

This creates a natural conversation flow where:
- **First message** replies to the user (or specified target)
- **Subsequent messages** appear as regular chat messages
- **Delays** make it feel like natural typing/thinking time

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

### Message Delay
The delay between messages is set to **1.5 seconds** by default. To change this, edit [action_executor.py](shin_ai/core/action_executor.py#L42):

```python
await asyncio.sleep(1.5)  # Change this value
```

Recommended ranges:
- **0.5 - 1.0 seconds**: Quick responses, less natural
- **1.5 - 2.0 seconds**: Natural conversation pace (recommended)
- **2.5 - 3.0 seconds**: Slower, more dramatic

### Reply Behavior
Currently, only the first message is a reply. To change this behavior, modify the logic in [action_executor.py](shin_ai/core/action_executor.py#L44-L48).

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
- [x] Delay between messages (for dramatic effect) ✅ **Implemented: 1.5s delay**
- [x] First message as reply, rest as normal messages ✅ **Implemented**
- [ ] Typing indicators between multi-messages
- [ ] Configurable delay per message
- [ ] Max message limit per response
- [ ] Message grouping/batching options
- [ ] Per-message media support (images in different messages)
- [ ] Smart delay based on message length

---

**Last Updated**: 2026-02-09
