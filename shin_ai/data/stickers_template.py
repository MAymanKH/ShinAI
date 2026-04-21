TELEGRAM_STICKERS = {
    "TG_CLOWN_STICKER": "placeholder",
    "TG_LAUGHING_STICKER": "placeholder",
}

TELEGRAM_STICKER_MAPPINGS = f"""
        - Clown: {TELEGRAM_STICKERS["TG_CLOWN_STICKER"]}
        - Laughing: {TELEGRAM_STICKERS["TG_LAUGHING_STICKER"]}
"""

TELEGRAM_STICKER_TO_DESCRIPTION = {
    TELEGRAM_STICKERS["TG_CLOWN_STICKER"]: "Clown / You are a clown",
    TELEGRAM_STICKERS["TG_LAUGHING_STICKER"]: "Laughing / Joy",
}

# WhatsApp stickers map to local file paths (relative to this file).
# For example, if you place "dog.webp" in the "shin_ai/data/whatsapp_stickers/" folder,
# you would set "WA_DOG_STICKER": "dog.webp".
WHATSAPP_STICKERS = {
    "WA_DOG_STICKER": "dog.webp",
    "WA_CAT_STICKER": "cat.webp",
}

WHATSAPP_STICKER_MAPPINGS = f"""
        - Dog: {WHATSAPP_STICKERS["WA_DOG_STICKER"]}
        - Cat: {WHATSAPP_STICKERS["WA_CAT_STICKER"]}
"""

WHATSAPP_STICKER_TO_DESCRIPTION = {
    WHATSAPP_STICKERS["WA_DOG_STICKER"]: "Dog / Happy dog",
    WHATSAPP_STICKERS["WA_CAT_STICKER"]: "Cat / Grumpy cat",
}
