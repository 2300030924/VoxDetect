import re

class PunctuationRestorer:
    """
    Simple rule-based punctuation + casing fixer.
    Enough for lightweight text cleanup.
    """

    def restore(self, text):
        if not text or text.strip() == "":
            return text

        text = text.strip()

        # Capitalize first letter
        text = text[0].upper() + text[1:]

        # Add period if missing
        if text[-1].isalnum():
            text += "."

        # Fix spacing
        text = re.sub(r"\s+", " ", text)

        return text
