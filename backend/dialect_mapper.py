"""
Egyptian Arabic Dialect Mapper
Maps MSA-like Arabic text to Egyptian colloquial variants:
- Cairene
- Sa'idi
- Alexandrian
- Egyptian Bedouin
"""

import json
import os
import re
from typing import Dict, List


# Safe shared MSA -> Egyptian colloquial replacements.
# Keep this list conservative to avoid semantic corruption.
COMMON_MSA_TO_EGYPTIAN: Dict[str, str] = {
    "ماذا": "ايه",
    "لماذا": "ليه",
    "كيف": "ازاي",
    "أين": "فين",
    "متى": "امتى",
    "الآن": "دلوقتي",
    "هذا": "ده",
    "هذه": "دي",
    "هؤلاء": "دول",
    "ذلك": "ده",
    "تلك": "دي",
    "ليس": "مش",
    "لست": "مش",
    "ليست": "مش",
    "أريد": "عايز",
    "أريد أن": "عايز",
    "لا أريد": "مش عايز",
    "أستطيع": "اقدر",
    "استطيع": "اقدر",
    "يمكنني": "اقدر",
    "لدي": "عندي",
    "عندي": "عندي",
    "معي": "معايا",
    "معك": "معاك",
    "شيء": "حاجة",
    "أشياء": "حاجات",
    "قليلا": "شوية",
    "قليل": "شوية",
    "كثيرا": "كتير",
    "كثير": "كتير",
    "حسنا": "تمام",
    "حسنًا": "تمام",
    "بالتأكيد": "اكيد",
    "مرحبا": "اهلا",
    "شكرا": "شكرا",
    "شكراً": "شكرا",
    "من فضلك": "لو سمحت",
    "معذرة": "معلش",
    "آسف": "اسف",
    "أنا": "انا",
    "أنت": "انت",
    "أنتِ": "انتي",
    "نحن": "احنا",
    "ذهب": "راح",
    "يذهب": "يروح",
    "جاء": "جه",
    "يجيء": "ييجي",
    "انظر": "بص",
    "ينظر": "يبص",
    "يتحدث": "يتكلم",
    "يتكلم": "يتكلم",
    "مشكلة": "مشكلة",
}


# Dialect-specific flavor lexicons (40-60 each).
# These are additive and intentionally phrase-heavy for stronger identity.
DIALECT_VOCAB: Dict[str, Dict[str, str]] = {
    "cairene": {
        "كيف حالك": "عامل ايه",
        "بخير": "تمام",
        "صديقي": "يا معلم",
        "يا صديقي": "يا باشا",
        "أفهم": "فاهم",
        "لا أفهم": "مش فاهم",
        "أحتاج": "محتاج",
        "ممتاز": "جامد",
        "رائع": "تحفة",
        "جيد": "كويس",
        "سيئ": "وحش",
        "جدا": "اوي",
        "حالا": "دلوقتي",
        "المنزل": "البيت",
        "العمل": "الشغل",
        "وظيفة": "شغل",
        "المساعدة": "المساعدة",
        "توضيح": "شرح",
        "إجابة": "رد",
        "سؤال": "سؤال",
        "الأمر": "الموضوع",
        "الأمور": "المواضيع",
        "فورا": "حالا",
        "حقيقي": "بجد",
        "حقا": "فعلا",
        "أحيانا": "ساعات",
        "غالبا": "غالبا",
        "دوما": "دايما",
        "سوف": "ه",
        "لن": "مش ه",
        "لم": "ما",
        "كما": "زي ما",
        "أيضا": "كمان",
        "الرجاء": "لو سمحت",
        "يسرني": "مبسوط اني",
        "أعتذر": "معلش",
        "حاليا": "دلوقتي",
        "إلى": "لـ",
        "من أجل": "عشان",
        "حيث": "في المكان اللي",
        "لهذا": "علشان كده",
        "تلك": "دي",
        "ذلك": "ده",
        "الذي": "اللي",
        "التي": "اللي",
        "يرجى": "يا ريت",
        "ينبغي": "لازم",
        "يجب": "لازم",
        "الآن": "دلوقتي",
    },
    "saidi": {
        "كيف حالك": "عامل ايه يا اخوي",
        "بخير": "زين",
        "صديقي": "يا اخوي",
        "يا صديقي": "يا اخوي",
        "أفهم": "فاهم",
        "لا أفهم": "مش فاهم",
        "أحتاج": "محتاج",
        "ممتاز": "زين قوي",
        "رائع": "حلو قوي",
        "جيد": "زين",
        "سيئ": "مش زين",
        "جدا": "قوي",
        "حالا": "دلوك",
        "المنزل": "البيت",
        "العمل": "الشغل",
        "وظيفة": "شغل",
        "المساعدة": "العون",
        "توضيح": "شرح",
        "إجابة": "رد",
        "سؤال": "سؤال",
        "الأمر": "الموضوع",
        "الأمور": "المواضيع",
        "فورا": "حالا",
        "حقيقي": "بالجد",
        "حقا": "فعلا",
        "أحيانا": "ساعات",
        "غالبا": "غالبا",
        "دوما": "دايما",
        "سوف": "ه",
        "لن": "مش ه",
        "لم": "ما",
        "كما": "زي ما",
        "أيضا": "كمان",
        "الرجاء": "لو سمحت",
        "يسرني": "مبسوط اني",
        "أعتذر": "معلش",
        "حاليا": "دلوك",
        "إلى": "لـ",
        "من أجل": "عشان",
        "لهذا": "عشان كده",
        "الذي": "اللي",
        "التي": "اللي",
        "يرجى": "يا ريت",
        "ينبغي": "لازم",
        "يجب": "لازم",
        "تعال": "تعالى",
        "هيا": "يلا",
        "الطعام": "الاكل",
        "المال": "الفلوس",
        "كبير": "كبير",
        "صغير": "صغير",
    },
    "alexandrian": {
        "كيف حالك": "عامل ايه يا باشا",
        "بخير": "تمام يا باشا",
        "صديقي": "يا باشا",
        "يا صديقي": "يا باشا",
        "أفهم": "فاهم",
        "لا أفهم": "مش فاهم",
        "أحتاج": "محتاج",
        "ممتاز": "عظمة",
        "رائع": "فشيخ",
        "جيد": "تمام",
        "سيئ": "وحش",
        "جدا": "اوي",
        "حالا": "دلوقتي",
        "المنزل": "البيت",
        "العمل": "الشغل",
        "وظيفة": "شغل",
        "المساعدة": "المساعدة",
        "توضيح": "توضيح",
        "إجابة": "رد",
        "سؤال": "سؤال",
        "الأمر": "الموضوع",
        "الأمور": "المواضيع",
        "فورا": "حالا",
        "حقيقي": "بجد",
        "حقا": "فعلا",
        "أحيانا": "ساعات",
        "غالبا": "غالبا",
        "دوما": "دايما",
        "سوف": "ه",
        "لن": "مش ه",
        "لم": "ما",
        "كما": "زي ما",
        "أيضا": "كمان",
        "الرجاء": "لو سمحت",
        "يسرني": "مبسوط اني",
        "أعتذر": "معلش",
        "حاليا": "دلوقتي",
        "إلى": "لـ",
        "من أجل": "عشان",
        "لهذا": "علشان كده",
        "الذي": "اللي",
        "التي": "اللي",
        "يرجى": "يا ريت",
        "ينبغي": "لازم",
        "يجب": "لازم",
        "بسرعة": "على السريع",
        "جدا جدا": "اوي اوي",
        "صحيح": "مظبوط",
        "انتهى": "خلص",
        "انتظر": "استنى",
    },
    "bedouin": {
        "كيف حالك": "اخبارك ايه يا خوي",
        "بخير": "تمام والحمد لله",
        "صديقي": "يا خوي",
        "يا صديقي": "يا خوي",
        "أفهم": "فاهم",
        "لا أفهم": "ماني فاهم",
        "أحتاج": "محتاج",
        "ممتاز": "طيب",
        "رائع": "زين",
        "جيد": "زين",
        "سيئ": "مو زين",
        "جدا": "مرة",
        "حالا": "هالحين",
        "المنزل": "البيت",
        "العمل": "الشغل",
        "وظيفة": "شغل",
        "المساعدة": "العون",
        "توضيح": "شرح",
        "إجابة": "رد",
        "سؤال": "سؤال",
        "الأمر": "الموضوع",
        "الأمور": "المواضيع",
        "فورا": "حالا",
        "حقيقي": "بالحق",
        "حقا": "صدق",
        "أحيانا": "مرات",
        "غالبا": "غالبا",
        "دوما": "دايم",
        "سوف": "بـ",
        "لن": "ما راح",
        "لم": "ما",
        "كما": "زي ما",
        "أيضا": "بعد",
        "الرجاء": "لو سمحت",
        "يسرني": "يسعدني",
        "أعتذر": "عذرك",
        "حاليا": "هالحين",
        "إلى": "لـ",
        "من أجل": "عشان",
        "لهذا": "علشان كده",
        "الذي": "اللي",
        "التي": "اللي",
        "يرجى": "يا ريت",
        "ينبغي": "لازم",
        "يجب": "لازم",
        "تعال": "تعال",
        "هيا": "يلا",
        "انتظر": "تمهل",
        "سريعا": "بسرعة",
        "المال": "الفلوس",
    },
}


DIALECT_GREETINGS: Dict[str, Dict[str, str]] = {
    "cairene": {
        "hello": "اهلا وسهلا! عامل ايه؟",
        "goodbye": "يلا سلام، اشوفك على خير",
        "thanks": "تسلم يا كبير",
        "welcome": "ولا يهمك، تحت امرك",
        "sorry": "معلش، حقك عليا",
    },
    "saidi": {
        "hello": "اهلا يا اخوي! عامل ايه؟",
        "goodbye": "مع السلامة يا اخوي",
        "thanks": "تسلم يا غالي",
        "welcome": "ولا يهمك يا اخوي",
        "sorry": "معلش يا اخوي",
    },
    "alexandrian": {
        "hello": "اهلا يا باشا! عامل ايه؟",
        "goodbye": "يلا سلام يا باشا",
        "thanks": "متشكر يا معلم",
        "welcome": "ولا يهمك يا باشا",
        "sorry": "معلش يا باشا",
    },
    "bedouin": {
        "hello": "هلا يا خوي! اخبارك ايه؟",
        "goodbye": "في امان الله يا خوي",
        "thanks": "يعطيك العافية",
        "welcome": "حياك الله يا خوي",
        "sorry": "عذرك يا خوي",
    },
}


DIALECT_PROSODY: Dict[str, Dict[str, str]] = {
    "cairene": {"rate": "1.02", "pitch": "+2%", "volume": "loud"},
    "saidi": {"rate": "0.95", "pitch": "-4%", "volume": "medium"},
    "alexandrian": {"rate": "1.04", "pitch": "+4%", "volume": "loud"},
    "bedouin": {"rate": "0.93", "pitch": "-3%", "volume": "medium"},
}


FORMAL_WORDS = {
    "الذي",
    "التي",
    "الذين",
    "اللاتي",
    "حيث",
    "إذ",
    "إلا",
    "لذلك",
    "وبالتالي",
    "يمكنني",
    "أستطيع",
    "استطيع",
    "سوف",
    "ينبغي",
    "يتعين",
    "يتوجب",
    "يرجى",
    "نرجو",
    "بناء",
    "عليه",
    "بالإضافة",
    "علاوة",
    "ذلك",
    "تلك",
    "هؤلاء",
}

FORMAL_PHRASES = [
    "كيف يمكنني مساعدتك",
    "يسعدني مساعدتك",
    "يرجى التوضيح",
    "من فضلك قم",
    "بإمكاني مساعدتك",
    "وفقًا لما ذكرت",
    "إذا رغبت",
    "على سبيل المثال",
]


DIALECT_STYLE_MARKERS: Dict[str, List[str]] = {
    "cairene": ["يا كبير", "يا معلم", "يا باشا"],
    "saidi": ["يا اخوي", "يا طيب", "يا غالي"],
    "alexandrian": ["يا باشا", "يا معلم", "يا نجم"],
    "bedouin": ["يا خوي", "يا طيب", "يا غالي"],
}


# Common English technical words written with Arabic letters.
# We normalize them to canonical English tokens so downstream intent/model logic
# can understand user text more reliably.
ARABICIZED_ENGLISH_PATTERNS = [
    # Web and app basics
    (r"ويب\s*سايت|ويبس?ايت|ويبس?يت|ويب\s*سيت", "website"),
    (r"ويب\s*اب|ويب\s*آب", "webapp"),
    (r"سايت", "site"),
    (r"ابليكيشن|ابلكيشن|ابليكشن", "application"),
    (r"داش\s*بورد|داشبورد", "dashboard"),
    (r"فرونت\s*اند|فرونتاند", "frontend"),
    (r"باك\s*اند|باك\s*إند|باك\s*ايند", "backend"),
    (r"فول\s*ستاك|فولستاك", "fullstack"),
    (r"يو\s*اي|يوآي", "ui"),
    (r"يو\s*اكس|يوآكس", "ux"),

    # API and data exchange
    (r"اي\s*بي\s*اي|ابي\s*اي", "api"),
    (r"اند\s*بوينت|إند\s*بوينت|ايند\s*بوينت|اندبوينت|ايندبوينت", "endpoint"),
    (r"ريكوست|ريكويست|ركوست", "request"),
    (r"ريسبونس|ريسپونس|رسپونس", "response"),
    (r"جيسون|جسون|جاسون", "json"),
    (r"يو\s*ار\s*ال|يوارال|يو\s*ار\s*إل", "url"),
    (r"اتش\s*تي\s*تي\s*بي\s*اس|اتشتيتيبياس", "https"),
    (r"اتش\s*تي\s*تي\s*بي|اتشتيتيبي", "http"),

    # Auth and account terms
    (r"ايميل|إيميل", "email"),
    (r"باس\s*ورد|باسورد", "password"),
    (r"يوزر\s*نيم|يوزرنيم", "username"),
    (r"لوج\s*ان|لوجين", "login"),
    (r"لوج\s*اوت|لوجاوت", "logout"),
    (r"ساين\s*اب|سايناب", "signup"),
    (r"بروفايل", "profile"),
    (r"يوزر", "user"),

    # Programming terms
    (r"فنكشن|فانكشن", "function"),
    (r"ميثود", "method"),
    (r"كلاس", "class"),
    (r"اوبجكت|اوبجيكت", "object"),
    (r"موديول", "module"),
    (r"باكدج|باكج", "package"),
    (r"فريم\s*ورك|فريمورك", "framework"),
    (r"لايبراري|لايبرري|لايبررى", "library"),
    (r"ريبو|ريبوزيتوري|ريبوسيتوري|ريبوستوري", "repository"),
    (r"برانش", "branch"),
    (r"كوميت|كميت", "commit"),
    (r"بول\s*ريكويست|بول\s*ريكوست", "pull request"),

    # Debugging and runtime issues
    (r"ايرور|إيرور|ايرر", "error"),
    (r"اكسبشن|اكسيبشن|إكسبشن", "exception"),
    (r"باج|بج", "bug"),
    (r"ديباج|ديبج", "debug"),
    (r"فيكس", "fix"),
    (r"اشيو|ايشو", "issue"),

    # Infra and delivery
    (r"سيرفر", "server"),
    (r"هوست", "host"),
    (r"دومين", "domain"),
    (r"ديبلوي|ديبلوى", "deploy"),
    (r"بيلد", "build"),
    (r"كونفيج|كونفج", "config"),
    (r"داتا\s*بيز", "database"),
    (r"داتا\s*سيت|داتاسيت", "dataset"),
    (r"موديل|مودل", "model"),

    # Files and links
    (r"ابلود", "upload"),
    (r"داونلود", "download"),
    (r"فايل", "file"),
    (r"فولدر", "folder"),
    (r"لينك", "link"),

    # LLM terms
    (r"برومبت", "prompt"),
    (r"توكن", "token"),
    (r"تشات\s*بوت|شات\s*بوت|تشاتبوت|شاتبوت", "chatbot"),
    (r"جي\s*بي\s*تي|جيبتي", "gpt"),

    # Customer-service arabicized spellings
    (r"ريفند", "refund"),
    (r"ريتيرن|ريترن", "return"),
    (r"ريبليسمنت|ربليسمنت", "replacement"),
    (r"تيكت|تيكيت", "ticket"),
    (r"اسكاليت|اسكليت", "escalate"),
    (r"فولو\s*اب|فولواب", "follow up"),
    (r"اوردر", "order"),
    (r"شيبينج", "shipping"),
    (r"ديليفري|دليفري", "delivery"),
    (r"تراكنج", "tracking"),
    (r"اوت\s*بي|او\s*تي\s*بي", "otp"),
    (r"فاوتشر", "voucher"),
    (r"برومو\s*كود|بروموكود", "promo code"),

    # Action words frequently used in requests
    (r"(?:ا|أ|إ)?هاندل|(?:ا|أ|إ)?هندل", "handle"),
]


# Customer-service specific dictionaries.
# Goal: understand both Arabic + English customer-service language,
# then normalize to Egyptian colloquial wording for better intent handling.
CUSTOMER_SERVICE_EN_TO_EGYPTIAN: Dict[str, str] = {
    "customer service": "خدمة العملاء",
    "support": "الدعم",
    "help center": "مركز المساعدة",
    "contact us": "كلمنا",
    "agent": "موظف خدمة عملاء",
    "representative": "موظف خدمة عملاء",
    "supervisor": "المشرف",
    "complaint": "شكوى",
    "feedback": "ملاحظات",
    "issue": "مشكلة",
    "problem": "مشكلة",
    "urgent": "ضروري",
    "priority": "اولوية",
    "ticket": "تذكرة دعم",
    "case": "حالة",
    "status": "الحالة",
    "pending": "لسه قيد المراجعة",
    "resolved": "اتحلت",
    "close ticket": "اقفل التذكرة",
    "escalate": "صعد المشكلة",
    "follow up": "متابعة",
    "callback": "رجعولي مكالمة",
    "response time": "وقت الرد",
    "sla": "مدة الخدمة المتفق عليها",
    "refund": "استرجاع فلوس",
    "return": "ارجاع",
    "replacement": "استبدال",
    "exchange": "تبديل",
    "cancel": "الغي",
    "cancellation": "الغاء",
    "invoice": "فاتورة",
    "receipt": "ايصال",
    "payment": "دفع",
    "failed payment": "الدفع فشل",
    "transaction": "عملية دفع",
    "charge": "خصم",
    "double charge": "اتخصم مرتين",
    "promo code": "كود خصم",
    "discount": "خصم",
    "voucher": "كوبون",
    "subscription": "اشتراك",
    "renewal": "تجديد",
    "order": "اوردر",
    "order number": "رقم الاوردر",
    "shipment": "شحنة",
    "shipping": "شحن",
    "delivery": "توصيل",
    "delay": "تأخير",
    "tracking": "تتبع",
    "tracking number": "رقم التتبع",
    "out for delivery": "نزل للتوصيل",
    "not delivered": "ماوصلش",
    "wrong item": "المنتج غلط",
    "damaged": "تالف",
    "missing item": "في حاجة ناقصة",
    "account": "الحساب",
    "login": "تسجيل دخول",
    "password reset": "اعادة تعيين الباسورد",
    "verification": "تأكيد",
    "otp": "كود التحقق",
    "blocked": "مقفول",
    "activate": "فعل",
    "deactivate": "وقف",
    "policy": "السياسة",
    "terms": "الشروط",
    "privacy": "الخصوصية",
}


CUSTOMER_SERVICE_AR_TO_EGYPTIAN: Dict[str, str] = {
    "خدمة الزبائن": "خدمة العملاء",
    "خدمة العملاء": "خدمة العملاء",
    "الدعم الفني": "الدعم",
    "الدعم التقني": "الدعم",
    "المساعدة": "المساعدة",
    "التواصل معنا": "كلمنا",
    "ممثل خدمة العملاء": "موظف خدمة عملاء",
    "الموظف المختص": "الموظف المسؤول",
    "المشرف": "المشرف",
    "شكوى": "شكوى",
    "بلاغ": "شكوى",
    "ملاحظة": "ملاحظة",
    "مشكلة": "مشكلة",
    "عطل": "مشكلة",
    "طارئ": "ضروري",
    "أولوية": "اولوية",
    "تذكرة": "تذكرة دعم",
    "رقم الحالة": "رقم الحالة",
    "حالة الطلب": "حالة الطلب",
    "قيد المعالجة": "لسه قيد المراجعة",
    "تم الحل": "اتحلت",
    "إغلاق التذكرة": "اقفل التذكرة",
    "تصعيد": "صعد المشكلة",
    "متابعة": "متابعة",
    "اتصال هاتفي": "مكالمة",
    "مدة الاستجابة": "وقت الرد",
    "استرجاع": "استرجاع فلوس",
    "استرداد": "استرجاع فلوس",
    "إرجاع": "ارجاع",
    "استبدال": "استبدال",
    "تبديل": "تبديل",
    "إلغاء": "الغاء",
    "فاتورة": "فاتورة",
    "إيصال": "ايصال",
    "الدفع": "دفع",
    "عملية الدفع": "عملية دفع",
    "خصم": "خصم",
    "خصم مكرر": "اتخصم مرتين",
    "رمز الخصم": "كود خصم",
    "كوبون": "كوبون",
    "اشتراك": "اشتراك",
    "تجديد": "تجديد",
    "طلب": "اوردر",
    "رقم الطلب": "رقم الاوردر",
    "شحنة": "شحنة",
    "الشحن": "شحن",
    "التوصيل": "توصيل",
    "تأخير": "تأخير",
    "تتبع": "تتبع",
    "رقم التتبع": "رقم التتبع",
    "خرج للتسليم": "نزل للتوصيل",
    "لم يتم التسليم": "ماوصلش",
    "منتج خاطئ": "المنتج غلط",
    "منتج تالف": "تالف",
    "عنصر مفقود": "في حاجة ناقصة",
    "الحساب": "الحساب",
    "تسجيل الدخول": "تسجيل دخول",
    "إعادة تعيين كلمة المرور": "اعادة تعيين الباسورد",
    "التحقق": "تأكيد",
    "رمز التحقق": "كود التحقق",
    "محظور": "مقفول",
    "مفعل": "متفعل",
    "غير مفعل": "مش متفعل",
    "سياسة": "السياسة",
    "الشروط": "الشروط",
    "الخصوصية": "الخصوصية",
}


_CUSTOM_CODESWITCH_CACHE = {
    "path": "",
    "mtime": None,
    "patterns": [],
}


_CODESWITCH_CORPUS_CACHE = {
    "path": "",
    "mtime": None,
    "lines": [],
}


def _resolve_custom_codeswitch_lexicon_path() -> str:
    """Resolve configurable path for custom code-switch lexicon JSON."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    configured = (os.getenv("CUSTOM_CODESWITCH_LEXICON_PATH", "") or "").strip()
    if not configured:
        return os.path.join(backend_dir, "custom_codeswitch_lexicon.json")

    if os.path.isabs(configured):
        return configured

    return os.path.abspath(os.path.join(backend_dir, configured))


def _load_custom_codeswitch_patterns() -> list[tuple[str, str, bool]]:
    """Load custom code-switch patterns from JSON with lightweight mtime cache."""
    path = _resolve_custom_codeswitch_lexicon_path()

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _CUSTOM_CODESWITCH_CACHE["path"] = path
        _CUSTOM_CODESWITCH_CACHE["mtime"] = None
        _CUSTOM_CODESWITCH_CACHE["patterns"] = []
        return []

    cached_path = _CUSTOM_CODESWITCH_CACHE.get("path")
    cached_mtime = _CUSTOM_CODESWITCH_CACHE.get("mtime")
    if cached_path == path and cached_mtime == mtime:
        return _CUSTOM_CODESWITCH_CACHE.get("patterns", [])

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        _CUSTOM_CODESWITCH_CACHE["path"] = path
        _CUSTOM_CODESWITCH_CACHE["mtime"] = mtime
        _CUSTOM_CODESWITCH_CACHE["patterns"] = []
        return []

    patterns: list[tuple[str, str, bool]] = []

    exact_map = payload.get("exact_phrases", {}) if isinstance(payload, dict) else {}
    if isinstance(exact_map, dict):
        for source, target in exact_map.items():
            src = _normalize_text(str(source))
            dst = _normalize_text(str(target))
            if src and dst:
                patterns.append((re.escape(src), dst, False))

    regex_list = payload.get("regex_phrases", []) if isinstance(payload, dict) else []
    if isinstance(regex_list, list):
        for item in regex_list:
            if not isinstance(item, dict):
                continue
            source_pattern = _normalize_text(str(item.get("pattern", "")))
            target = _normalize_text(str(item.get("replacement", "")))
            if source_pattern and target:
                patterns.append((source_pattern, target, True))

    _CUSTOM_CODESWITCH_CACHE["path"] = path
    _CUSTOM_CODESWITCH_CACHE["mtime"] = mtime
    _CUSTOM_CODESWITCH_CACHE["patterns"] = patterns
    return patterns


def _resolve_codeswitch_corpus_path() -> str:
    """Resolve path for external mixed-language dataset samples."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    configured = (os.getenv("CODESWITCH_DATASET_PATH", "") or "").strip()
    if not configured:
        return os.path.join(backend_dir, "codeswitch_dataset_samples.txt")

    if os.path.isabs(configured):
        return configured

    return os.path.abspath(os.path.join(backend_dir, configured))


def _load_codeswitch_corpus_lines() -> list[str]:
    """Load code-switch dataset lines with mtime caching."""
    path = _resolve_codeswitch_corpus_path()

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _CODESWITCH_CORPUS_CACHE["path"] = path
        _CODESWITCH_CORPUS_CACHE["mtime"] = None
        _CODESWITCH_CORPUS_CACHE["lines"] = []
        return []

    if (
        _CODESWITCH_CORPUS_CACHE.get("path") == path
        and _CODESWITCH_CORPUS_CACHE.get("mtime") == mtime
    ):
        return _CODESWITCH_CORPUS_CACHE.get("lines", [])

    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = _normalize_text(raw_line)
                if line and not line.startswith("#"):
                    lines.append(line)
    except OSError:
        lines = []

    _CODESWITCH_CORPUS_CACHE["path"] = path
    _CODESWITCH_CORPUS_CACHE["mtime"] = mtime
    _CODESWITCH_CORPUS_CACHE["lines"] = lines
    return lines


def _extract_latin_tokens(text: str) -> set[str]:
    """Extract normalized latin tokens used in code-switched Arabic speech."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-\./']*", text or "")
    return {t.lower() for t in tokens if len(t) >= 2}


def get_codeswitch_context_hints(text: str, max_hints: int = 3) -> list[str]:
    """Return nearest corpus examples to help model interpret mixed Arabic/English text."""
    query = _normalize_text(text)
    if not query:
        return []

    query_tokens = _extract_latin_tokens(query)
    if not query_tokens:
        return []

    lines = _load_codeswitch_corpus_lines()
    if not lines:
        return []

    scored: list[tuple[float, str]] = []
    query_lower = query.lower()

    for line in lines[:6000]:
        line_tokens = _extract_latin_tokens(line)
        if not line_tokens:
            continue

        overlap = query_tokens.intersection(line_tokens)
        if not overlap:
            continue

        jaccard = len(overlap) / max(1, len(query_tokens.union(line_tokens)))
        has_direct_phrase = 1.0 if any(tok in line.lower() and tok in query_lower for tok in overlap) else 0.0
        score = (len(overlap) * 0.7) + (jaccard * 0.3) + has_direct_phrase

        if score > 0:
            scored.append((score, line))

    scored.sort(key=lambda item: item[0], reverse=True)

    hints: list[str] = []
    for _, line in scored:
        if line not in hints:
            hints.append(line)
        if len(hints) >= max_hints:
            break

    return hints


def _normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def normalize_arabicized_english(text: str) -> str:
    """Normalize Arabic-script English terms to canonical English words."""
    result = _normalize_text(text)
    if not result:
        return result

    word_char = r"[\w\u0600-\u06FF]"
    for raw_pattern, replacement in ARABICIZED_ENGLISH_PATTERNS:
        pattern = re.compile(
            rf"(?<!{word_char})(?:{raw_pattern})(?!{word_char})",
            flags=re.UNICODE | re.IGNORECASE,
        )
        result = pattern.sub(replacement, result)

    return _normalize_text(result)


def normalize_customer_service_terms(text: str) -> str:
    """Normalize customer-service terms (EN + AR) to Egyptian colloquial phrasing."""
    result = _normalize_text(text)
    if not result:
        return result

    # English terms (latin script) with case-insensitive boundaries.
    en_sorted = sorted(
        CUSTOMER_SERVICE_EN_TO_EGYPTIAN.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for english_term, egyptian_term in en_sorted:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(english_term)}(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        result = pattern.sub(egyptian_term, result)

    # Arabic terms with safer whole-word boundaries.
    ar_sorted = sorted(
        CUSTOMER_SERVICE_AR_TO_EGYPTIAN.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    word_char = r"[\w\u0600-\u06FF]"
    for arabic_term, egyptian_term in ar_sorted:
        pattern = re.compile(
            rf"(?<!{word_char}){re.escape(arabic_term)}(?!{word_char})",
            flags=re.UNICODE,
        )
        result = pattern.sub(egyptian_term, result)

    return _normalize_text(result)


def normalize_custom_codeswitch_terms(text: str) -> str:
    """Apply user-customizable code-switch lexicon from JSON file."""
    result = _normalize_text(text)
    if not result:
        return result

    patterns = _load_custom_codeswitch_patterns()
    if not patterns:
        return result

    word_char = r"[\w\u0600-\u06FF]"
    for raw_pattern, replacement, is_regex in patterns:
        try:
            if is_regex:
                pattern = re.compile(raw_pattern, flags=re.UNICODE | re.IGNORECASE)
            else:
                pattern = re.compile(
                    rf"(?<!{word_char}){raw_pattern}(?!{word_char})",
                    flags=re.UNICODE | re.IGNORECASE,
                )
            result = pattern.sub(replacement, result)
        except re.error:
            continue

    return _normalize_text(result)


def normalize_codeswitch_text(text: str) -> str:
    """Unified normalization for mixed Arabic/English Egyptian code-switching."""
    result = normalize_arabicized_english(text)
    result = normalize_customer_service_terms(result)
    result = normalize_custom_codeswitch_terms(result)
    return _normalize_text(result)


def calculate_formality_score(text: str) -> float:
    """Return formality score (0-100). Higher means more MSA/formal style."""
    normalized = _normalize_text(text)
    if not normalized:
        return 0.0

    words = re.findall(r"[\u0600-\u06FF]+", normalized)
    if not words:
        return 0.0

    formal_hits = sum(1 for w in words if w in FORMAL_WORDS)
    lexical_score = (formal_hits / max(1, len(words))) * 100.0

    phrase_bonus = 0.0
    for phrase in FORMAL_PHRASES:
        if phrase in normalized:
            phrase_bonus += 15.0

    structure_bonus = 0.0
    if re.search(r"[ًٌٍَُِّْ]", normalized):
        structure_bonus += 8.0
    if "؛" in normalized or ";" in normalized:
        structure_bonus += 6.0
    if re.search(r"\b(سوف|لن|قد|إن)\b", normalized):
        structure_bonus += 10.0

    score = min(100.0, lexical_score + phrase_bonus + structure_bonus)
    return score


def is_response_too_formal(text: str, max_formality: float = 28.0) -> bool:
    """Return True when response style is too formal for Egyptian colloquial UX."""
    if len(_normalize_text(text)) < 5:
        return False
    return calculate_formality_score(text) > max_formality


def strengthen_colloquial_enforcement(text: str, dialect: str = "cairene") -> str:
    """Convert common formal phrases to colloquial before dialect mapping."""
    result = _normalize_text(text)

    common_patterns = {
        r"\bكيف يمكنني مساعدتك\b": "اقدر اساعدك بايه",
        r"\bيمكنني مساعدتك\b": "اقدر اساعدك",
        r"\bيسعدني مساعدتك\b": "مبسوط اني اساعدك",
        r"\bيرجى\b": "يا ريت",
        r"\bمن فضلك\b": "لو سمحت",
        r"\bأعتذر\b": "معلش",
        r"\bسوف\b": "ه",
        r"\bلن\b": "مش ه",
        r"\bيجب\b": "لازم",
        r"\bينبغي\b": "لازم",
        r"\bذلك\b": "ده",
        r"\bتلك\b": "دي",
        r"\bهؤلاء\b": "دول",
        r"\bالذي\b": "اللي",
        r"\bالتي\b": "اللي",
    }

    dialect_patterns = {
        "cairene": {
            r"\bاقدر\b": "اقدر",
            r"\bدلوقتي\b": "دلوقتي",
        },
        "saidi": {
            r"\bدلوقتي\b": "دلوك",
            r"\bيا باشا\b": "يا اخوي",
        },
        "alexandrian": {
            r"\bيا اخوي\b": "يا باشا",
            r"\bاوي\b": "اوي",
        },
        "bedouin": {
            r"\bدلوقتي\b": "هالحين",
            r"\bيا باشا\b": "يا خوي",
        },
    }

    for pattern, replacement in common_patterns.items():
        result = re.sub(pattern, replacement, result, flags=re.UNICODE)

    for pattern, replacement in dialect_patterns.get(dialect, {}).items():
        result = re.sub(pattern, replacement, result, flags=re.UNICODE)

    return _normalize_text(result)


def transform_to_dialect(text: str, dialect: str) -> str:
    """Transform text into selected Egyptian dialect with safe whole-word replacement."""
    normalized = _normalize_text(text)
    if not normalized:
        return normalized

    if dialect not in DIALECT_VOCAB:
        return normalized

    vocab = {**COMMON_MSA_TO_EGYPTIAN, **DIALECT_VOCAB[dialect]}
    sorted_mappings = sorted(vocab.items(), key=lambda item: len(item[0]), reverse=True)

    result = normalized
    word_char = r"[\w\u0600-\u06FF]"
    for standard, dialectal in sorted_mappings:
        pattern = re.compile(
            rf"(?<!{word_char}){re.escape(standard)}(?!{word_char})",
            flags=re.UNICODE,
        )
        result = pattern.sub(dialectal, result)

    # Final cleanup
    result = re.sub(r"\s+", " ", result, flags=re.UNICODE).strip()
    result = re.sub(r"\s+([؟!.,،])", r"\1", result, flags=re.UNICODE)
    return result


def get_dialect_prosody(dialect: str) -> Dict[str, str]:
    """Get SSML prosody parameters for a dialect."""
    return DIALECT_PROSODY.get(dialect, DIALECT_PROSODY["cairene"])


def get_greeting(dialect: str, greeting_type: str = "hello") -> str:
    """Get a dialect-specific greeting."""
    greetings = DIALECT_GREETINGS.get(dialect, DIALECT_GREETINGS["cairene"])
    return greetings.get(greeting_type, greetings["hello"])


def get_available_dialects() -> List[Dict[str, str]]:
    """Return list of available dialects with metadata."""
    return [
        {
            "id": "cairene",
            "name_ar": "قاهرية",
            "name_en": "Cairene (Cairo)",
            "description": "اللهجة القاهرية - لهجة القاهرة والدلتا",
        },
        {
            "id": "saidi",
            "name_ar": "صعيدية",
            "name_en": "Saidi (Upper Egyptian)",
            "description": "اللهجة الصعيدية - لهجة صعيد مصر",
        },
        {
            "id": "alexandrian",
            "name_ar": "اسكندرانية",
            "name_en": "Alexandrian",
            "description": "اللهجة الاسكندرانية - لهجة الاسكندرية",
        },
        {
            "id": "bedouin",
            "name_ar": "بدوية",
            "name_en": "Bedouin (Egyptian)",
            "description": "اللهجة البدوية المصرية - لهجة سيناء والصحرا",
        },
    ]
