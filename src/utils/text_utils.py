import unicodedata


def norm_text(text):
    return unicodedata.normalize("NFKC", text)
