"""Transliteration engines:
1. Urdu (Nastaliq / Perso-Arabic) -> Roman Urdu (high-coverage conversational lexicon + phonetic ligature engine)
2. Hindi (Devanagari) -> Roman Hindi with LIPI (abhishekgautamm/lipi)
"""
import re
import string
import streamlit as st

LIPI_MODEL_ID = "abhishekgautamm/lipi"
BACKENDS = {
    "Lexicon + model (recommended)": "map_lipi",
    "Model only": "lipi",
}

_DEV = r"\u0900-\u0963\u0966-\u097f"
_DEV_RUN = re.compile(f"[{_DEV}]+(?: +[{_DEV}]+)*")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_URDU = r"\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
_URDU_CHAR_RE = re.compile(f"[{_URDU}]")

URDU_PUNCTUATION = {
    "؟": "?",
    "،": ",",
    "۔": ".",
    "؛": ";",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}

ALL_PUNCTUATION = set(string.punctuation + "".join(URDU_PUNCTUATION.keys()) + "،۔؟؛“”‘’")

# High-frequency conversational Urdu to Roman Urdu lexicon
URDU_LEXICON = {
    "آپ": "aap",
    "اپ": "aap",
    "ہم": "hum",
    "وہ": "woh",
    "یہ": "yeh",
    "میں": "main",
    "تم": "tum",
    "تو": "tu",
    "ہے": "hai",
    "ہیں": "hain",
    "تھا": "tha",
    "تھی": "thi",
    "تھے": "thay",
    "تھیں": "theen",
    "ہوں": "hoon",
    "ہو": "ho",
    "ہوگا": "hoga",
    "ہوگی": "hogi",
    "ہوںگے": "honge",
    "ہوں گے": "honge",
    "ہوئے": "huye",
    "ہوا": "hua",
    "ہوئی": "hui",
    "ہوتا": "hota",
    "ہوتی": "hoti",
    "ہوتے": "hote",
    "نہیں": "nahi",
    "نہی": "nahi",
    "نہ": "na",
    "کیا": "kya",
    "کیوں": "kyun",
    "کیسے": "kaise",
    "کیسی": "kaisi",
    "کیسا": "kaisa",
    "کب": "kab",
    "کہاں": "kahan",
    "کون": "kaun",
    "کتنا": "kitna",
    "کتنی": "kitni",
    "کتنے": "kitne",
    "کوئی": "koi",
    "کچھ": "kuch",
    "سب": "sab",
    "بہت": "bahut",
    "زیادہ": "zyada",
    "کم": "kam",
    "ایک": "ek",
    "دو": "do",
    "تین": "teen",
    "چار": "chaar",
    "پانچ": "paanch",
    "اچھا": "accha",
    "اچھی": "acchi",
    "اچھے": "acche",
    "ٹھیک": "theek",
    "شکریہ": "shukriya",
    "مہربانی": "meherbani",
    "سلام": "salam",
    "وعلیکم": "walaikum",
    "خدا": "khuda",
    "حافظ": "hafiz",
    "الحمدللہ": "alhamdulillah",
    "انشاءاللہ": "inshallah",
    "ماشاءاللہ": "mashallah",
    "خیریت": "khairiyat",
    "بات": "baat",
    "باتیں": "baatein",
    "کام": "kaam",
    "وقت": "waqt",
    "طرح": "tarah",
    "لوگ": "log",
    "لوگوں": "logon",
    "چیز": "cheez",
    "چیزیں": "cheezein",
    "کرنا": "karna",
    "کرنے": "karne",
    "کرنی": "karni",
    "کرتا": "karta",
    "کرتی": "karti",
    "کرتے": "karte",
    "کریں": "karein",
    "کریںگے": "karenge",
    "کریں گے": "karenge",
    "کرو": "karo",
    "کر": "kar",
    "کیا": "kiya",
    "کیے": "kiye",
    "کہا": "kaha",
    "کہہ": "keh",
    "کہتے": "kehte",
    "کہتی": "kehti",
    "کہتا": "kehta",
    "کہیں": "kahein",
    "رہا": "raha",
    "رہی": "rahi",
    "رہے": "rahe",
    "رہیں": "rahein",
    "رہنا": "rehna",
    "جانا": "jaana",
    "جاتے": "jaate",
    "جاتی": "jaati",
    "جاتا": "jaata",
    "جائیں": "jaayein",
    "جاؤ": "jao",
    "گیا": "gaya",
    "گئی": "gayi",
    "گئے": "gaye",
    "آنا": "aana",
    "آتا": "aata",
    "آتی": "aati",
    "آتے": "aate",
    "آئیں": "aayein",
    "آیا": "aaya",
    "آئی": "aayi",
    "آئے": "aaye",
    "دینا": "dena",
    "دیتے": "dete",
    "دیتی": "deti",
    "دیں": "dein",
    "دیا": "diya",
    "دی": "di",
    "لیتے": "lete",
    "لیتی": "leti",
    "لیں": "lein",
    "لیا": "liya",
    "لیکن": "lekin",
    "اگر": "agar",
    "مگر": "magar",
    "کیونکہ": "kyunki",
    "اس": "is",
    "ان": "un",
    "اسکا": "iska",
    "اسکی": "iski",
    "اسکے": "iske",
    "اسکےلیے": "iske liye",
    "اس لیے": "is liye",
    "اسلیے": "isliye",
    "انکا": "unka",
    "انکی": "unki",
    "انکے": "unke",
    "انکےلیے": "unke liye",
    "میرا": "mera",
    "میری": "meri",
    "میرے": "mere",
    "تمہارا": "tumhara",
    "تمہاری": "tumhari",
    "تمہارے": "tumhare",
    "ہمارا": "hamara",
    "ہماری": "hamari",
    "ہمارے": "hamare",
    "اپنا": "apna",
    "अपनी": "apni",
    "اپنے": "apne",
    "مجھے": "mujhe",
    "تجھے": "tujhe",
    "ہمیں": "humein",
    "انہیں": "unhein",
    "انہو": "unho",
    "انہوں": "unhon",
    "اسنے": "usne",
    "اس نے": "us ne",
    "کا": "ka",
    "کی": "ki",
    "کے": "ke",
    "کو": "ko",
    "سے": "se",
    "پر": "par",
    "تک": "tak",
    "اور": "aur",
    "یا": "ya",
    "بھی": "bhi",
    "تو": "to",
    "ابھی": "abhi",
    "کبھی": "kabhi",
    "جب": "jab",
    "تب": "tab",
    "اب": "ab",
    "یہاں": "yahan",
    "وہاں": "wahan",
    "جہاں": "jahan",
    "پہلے": "pehlay",
    "بعد": "baad",
    "ساتھ": "saath",
    "پاس": "paas",
    "شروع": "shuru",
    "ختم": "khatam",
    "دیکھنا": "dekhna",
    "دیکھ": "dekh",
    "دیکھا": "dekha",
    "سوچ": "soch",
    "سمجھ": "samajh",
    "سمجھنا": "samajhna",
    "سمجھتے": "samajhte",
    "بتانا": "batana",
    "بتائیں": "batayein",
    "سننا": "sunna",
    "سنا": "suna",
    "پڑھنا": "parhna",
    "لکھنا": "likhna",
    "ویڈیو": "video",
    "آڈیو": "audio",
    "سسٹم": "system",
    "طریقہ": "tareeqa",
    "سوال": "sawal",
    "جواب": "jawab",
    "مسئلہ": "masla",
    "حل": "hal",
    "ضرور": "zaroor",
    "ضرورت": "zaroorat",
    "شاید": "shayad",
    "بالکل": "bilkul",
    "صحیح": "sahi",
    "غلط": "ghalat",
    "نیا": "naya",
    "نئی": "nayi",
    "نئے": "naye",
    "پرانا": "purana",
    "آج": "aaj",
    "کل": "kal",
    "پرسوں": "parson",
    "صبح": "subah",
    "شام": "shaam",
    "رات": "raat",
    "دن": "din",
    "سال": "saal",
    "مہینہ": "maheena",
    "ہفتہ": "hafta",
    "دوبارہ": "dobara",
    "صرف": "sirf",
    "ہمیشہ": "hamesha",
    "والا": "wala",
    "والی": "wali",
    "والے": "wale",
    "دوست": "dost",
    "بھائی": "bhai",
    "بہن": "behan",
    "گھر": "ghar",
    "دفتر": "daftar",
    "پیسے": "paise",
    "رقم": "raqam",
    "نمبر": "number",
    "پاکستان": "pakistan",
    "اردو": "urdu",
    "انگریزی": "angrezi",
    "انگلش": "english",
}

# Aspirated consonant clusters in Urdu (Do Chashmi He combinations)
URDU_ASPIRATES = [
    ("بھ", "bh"),
    ("پھ", "ph"),
    ("تھ", "th"),
    ("ٹھ", "th"),
    ("جھ", "jh"),
    ("چھ", "chh"),
    ("دھ", "dh"),
    ("ڈھ", "dh"),
    ("رھ", "rh"),
    ("ڑھ", "rh"),
    ("کھ", "kh"),
    ("گھ", "gh"),
    ("لھ", "lh"),
    ("مھ", "mh"),
    ("نھ", "nh"),
    ("وھ", "wh"),
    ("یھ", "yh"),
]

URDU_CHAR_MAP = {
    "آ": "aa",
    "ا": "a",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "o",
    "ہ": "h",
    "ھ": "h",
    "ء": "'",
    "ی": "i",
    "ے": "e",
    "ئ": "i",
    "ؤ": "o",
    "ۃ": "t",
    "ۂ": "h",
    "ي": "i",
    "ك": "k",
}


def _transliterate_urdu_word(word: str) -> str:
    """Convert an individual Urdu word using lexicon lookup or phonetic ligature mapping."""
    clean_word = word.strip()
    if not clean_word:
        return ""

    # Strip any leading and trailing punctuation (both Urdu and ASCII)
    prefix, suffix = "", ""
    while clean_word and clean_word[0] in ALL_PUNCTUATION:
        c = clean_word[0]
        prefix += URDU_PUNCTUATION.get(c, c)
        clean_word = clean_word[1:]
    while clean_word and clean_word[-1] in ALL_PUNCTUATION:
        c = clean_word[-1]
        suffix = URDU_PUNCTUATION.get(c, c) + suffix
        clean_word = clean_word[:-1]

    if not clean_word:
        return prefix + suffix

    # Direct lexicon match
    if clean_word in URDU_LEXICON:
        return prefix + URDU_LEXICON[clean_word] + suffix

    # Phonetic decomposition for non-lexicon words
    w = clean_word
    for urdu_pair, roman_pair in URDU_ASPIRATES:
        w = w.replace(urdu_pair, roman_pair)

    res = []
    i = 0
    n = len(w)
    while i < n:
        c = w[i]
        if c in URDU_CHAR_MAP:
            if c == "و":
                if i == 0:
                    res.append("w")
                elif i == n - 1:
                    res.append("o")
                else:
                    res.append("oo" if (i > 0 and w[i - 1] in ("ب", "پ", "ت", "ک", "م")) else "o")
            elif c == "ی":
                if i == 0:
                    res.append("y")
                elif i == n - 1:
                    res.append("i")
                else:
                    res.append("ee")
            elif c == "ے":
                res.append("e")
            elif c == "ہ":
                if i == n - 1 and len(w) > 2:
                    res.append("a")
                else:
                    res.append("h")
            elif c == "ا":
                res.append("a")
            else:
                res.append(URDU_CHAR_MAP[c])
        elif c in URDU_PUNCTUATION:
            res.append(URDU_PUNCTUATION[c])
        else:
            res.append(c)
        i += 1

    romanized = "".join(res)
    return prefix + romanized + suffix


def romanize_urdu(text: str) -> str:
    """Transliterate Urdu Nastaliq text to clean, readable Roman Urdu."""
    if not text:
        return ""

    # Replace Urdu punctuation first
    for u_p, r_p in URDU_PUNCTUATION.items():
        text = text.replace(u_p, r_p)

    tokens = re.split(r"(\s+)", text)
    result = []
    for token in tokens:
        if not token:
            continue
        if re.search(_URDU_CHAR_RE, token):
            result.append(_transliterate_urdu_word(token))
        else:
            result.append(token)

    joined = "".join(result)
    # Capitalize first letter of sentences
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), joined)


@st.cache_resource(show_spinner="Loading LIPI (first run downloads the model, ~45 MB)...")
def load_lipi(backend):
    from huggingface_hub import snapshot_download
    from lipi import Transliterator

    model_dir = snapshot_download(LIPI_MODEL_ID)
    if backend == "map_lipi":
        return Transliterator(
            backend="map_lipi", model_dir=model_dir, map_path=f"{model_dir}/lipi_map.json"
        )
    return Transliterator(model_dir=model_dir)


def romanize_hindi(text, backend="map_lipi"):
    t = load_lipi(backend)
    out = _DEV_RUN.sub(lambda m: t.transliterate_text(m.group()), text)
    out = out.replace("।", ".").replace("॥", ".").translate(_DEV_DIGITS)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)


def transliterate_to_roman(text: str, script: str = "urdu", lipi_backend: str = "map_lipi") -> str:
    """Master transliteration dispatcher: converts Urdu or Devanagari text to Roman script."""
    if not text:
        return ""
    has_urdu = bool(re.search(_URDU_CHAR_RE, text))
    has_dev = bool(re.search(f"[{_DEV}]", text))

    if script == "urdu" or (has_urdu and not has_dev):
        return romanize_urdu(text)
    elif script == "devanagari" or (has_dev and not has_urdu):
        return romanize_hindi(text, lipi_backend)
    elif has_urdu and has_dev:
        intermediate = romanize_hindi(text, lipi_backend)
        return romanize_urdu(intermediate)
    return text
