# ===========================================
# Members Configuration Template
# ===========================================
# This file contains example member data for the bot's social context feature.
# Copy this file to members.py and customize with your own group members.
#
# Each member entry uses their username (lowercase, without @) as the key.
# If a member doesn't have a username, use their Telegram user ID as a string.
#
# Fields:
#   - names: List of names/aliases to recognize this person (include @username)
#   - trigger_keywords: Words that indicate the user is being discussed
#   - preferred_name: How the bot should refer to this person
#   - role: Their relationship to the bot/group
#   - backstory: Optional background info for personality/context
#   - location: Optional location info

MEMBERS = {
    # Example: Bot creator
    "yourusername": {
        "names": ["@yourusername", "YourName", "YourNickname"],
        "trigger_keywords": ["creator", "father", "owner", "admin"],
        "preferred_name": "YourName",
        "role": "Group member. Bot creator.",
        "backstory": "The person who created and maintains this bot.",
        "location": "Your City"
    },
    
    # Example: Regular group member
    "friendusername": {
        "names": ["@friendusername", "FriendName"],
        "trigger_keywords": [],
        "preferred_name": "FriendName",
        "role": "Group member.",
        "backstory": "",
        "location": ""
    },
    
    # Example: Member without username (use user ID)
    # "1234567890": {
    #     "names": ["Nickname"],
    #     "trigger_keywords": [],
    #     "preferred_name": "Nickname",
    #     "role": "Group member.",
    #     "backstory": "",
    #     "location": ""
    # },
}
