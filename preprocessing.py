"""
Text preprocessing for fake product review detection.

Steps:
1. Handle empty / non-text input
2. Lowercase
3. Strip HTML tags and URLs
4. Expand common contractions so "don't" becomes "do not"
5. Keep letters, numbers, and spaces
6. Remove stopwords, but keep negation words (not, no, never, ...)
"""

import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
        STOPWORDS = set(stopwords.words("english"))
except Exception:
    word_tokenize = lambda text: text.split()
    STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "only", "own", "same", "so", "than", "too", "very",
        "s", "t", "can", "will", "just", "should", "now",
    }

# Negation and similar words that change the meaning of a review.
# These must stay in the text so TF-IDF can use them.
NEGATION_WORDS = {
    "no", "not", "never", "nor", "none", "nobody", "nothing", "neither",
    "nowhere", "cannot", "cant", "dont", "doesnt", "isnt", "wasnt",
    "werent", "hasnt", "havent", "hadnt", "wont", "wouldnt", "couldnt",
    "shouldnt", "without",
}

STOPWORDS = STOPWORDS - NEGATION_WORDS

# Turn common contractions into separate words before punctuation is removed.
CONTRACTIONS = [
    (re.compile(r"won't", re.I), " will not "),
    (re.compile(r"can't", re.I), " can not "),
    (re.compile(r"n't", re.I), " not "),
    (re.compile(r"'re", re.I), " are "),
    (re.compile(r"'s", re.I), " is "),
    (re.compile(r"'d", re.I), " would "),
    (re.compile(r"'ll", re.I), " will "),
    (re.compile(r"'ve", re.I), " have "),
    (re.compile(r"'m", re.I), " am "),
]


def clean_text(text) -> str:
    """Return a normalized review string ready for TF-IDF."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)

    # Keep letters, numbers, and spaces. Reviews often mention sizes, ratings, etc.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    kept = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in STOPWORDS:
            continue
        # Keep short negation words like "no"; drop other 1-letter tokens.
        if len(token) == 1 and token not in NEGATION_WORDS:
            continue
        kept.append(token)

    return " ".join(kept)
