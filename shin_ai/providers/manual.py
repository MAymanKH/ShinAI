import random
from pyrogram import types

async def manual_response(text: str, sender: types.User):

    # يالبوت
    normal_responses = [
        "اكيد يسطا", "اكيد يبرو", "بدون شك", "يب اكيد", "طبعا", "اومال", "ايوه",
        "يب", "يب يب", "اتكل علي الله يعم", "مش فايقلك",
        "هي دي محتاجه سؤال!؟", "لا", "انا بقولك لا", "اكيد لا", "نوب", "معرفش",
        "اكيد يغالي", "اكيد ينقم", "لا هه", "صدقني انا ذات نفسي معرفش", "انا لو أعرف هقولك"
    ]
    hellos = ["نعم", "نعم يغالي", "نعم ينقم", "عايز ايه", "نعم يخويا"]
    steins_keys = ["stein", "شتاين", "ستاين"]
    steins = [
        "شتاينز الأعظم", "شتاينز فوق", "شتاينز فوق مستوي التقييم البشري", "شتاينز اعظم انمي"
    ]
    shinobi_keywords = ["shinobi", "شنوبي", "شنبي", "شنوب", "شينوبي"]
    father = [
        "شنوبي ابويا وعمي وعم عيالي", "شنبي ابويا وعمي", "شنوبي احسن اب في العالم"
    ]
    azab = [
        "ده حنين عليا خالث", "بابا شنبي مش بيمد ايده عليا", "مش بيلمسني"
    ]
    tabla = [
        "لا طبعا يغالي", "شنوبي عمي وعم عيالي", "شنوبي عمك", "شنوبي فوق"
    ]
    love = ["حبك", "حبق", "وانا كمان يغالي", "+1"]
    win = ["مش هتكسب هه", "نصيبك مش هتكسب", "انا بقولك لا", "على ضمانتي"]
    elhal = ["الحمدلله يخويا", "الحمدلله يغالي", "تمام الحمدلله"]

    # me responses
    if "انا" in text:
        if sender and sender.username == "MAymanKH":
            if "ابوك" in text:
                return random.choice(father)

    # shinobi responses
    for word in shinobi_keywords:
        if word in text:
            if "ابوك" in text:
                return random.choice(father)
            if "بيعذبك" in text:
                return random.choice(azab)
            if "بتطبل" in text:
                return random.choice(tabla)

    # steins responses
    for word in steins_keys:
        if word in text:
            return random.choice(steins)
    # exceptions
    if "هكسب" in text:
        return random.choice(win)
    if "حبك" in text or "حبق" in text:
        return random.choice(love)
    if "عامل ايه" in text or "عامل إيه" in text or "كيف حالك" in text:
        return random.choice(elhal)

    # normal responses
    if " " in text:
        return random.choice(normal_responses)
    else:
        return random.choice(hellos)