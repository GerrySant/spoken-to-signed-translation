import re
from typing import Callable, List


def preprocess_lower_strip(s: str) -> str:
    """Lowercase and strip whitespace."""
    return str(s).strip().lower()


def preprocess_rule_based(s: str) -> str:
    """Remove '+' and trailing '-ix'."""
    s = s.replace("+", "")
    if s.endswith("-ix"):
        s = s[:-3]
    return s


def preprocess_spacylemma(s: str) -> str:
    """Remove SpaCy-style '--' artifacts."""
    return s.replace("--", "")


def preprocess_integer_with_punctuation(s: str) -> str:
    """
    Normalize standalone integer tokens while preserving:
    - decimals
    - alphanumeric tokens
    """
    s = s.strip()

    # 1) Keep decimals untouched
    if re.fullmatch(r"\d+[.,]\d+", s):
        return s

    # 2) Reject alphanumeric tokens (letters touching digits)
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return s

    # 3) Match pure integer with optional non-letter punctuation around it
    m = re.fullmatch(r"[^A-Za-z0-9]*([0-9]+)[^A-Za-z0-9]*", s)
    if m:
        return m.group(1)

    return s


def preprocess_keep_letters_only(s: str) -> str:
    """Keep only Unicode letters."""
    return "".join(ch for ch in s if ch.isalpha())


def get_progressive_gloss_normalizers() -> List[Callable[[str], str]]:
    """
    Returns the ordered list of progressive gloss normalization steps.
    The first entry (None) corresponds to the original gloss.
    """
    return [
        None,  # original gloss
        preprocess_lower_strip,
        preprocess_rule_based,
        preprocess_spacylemma,
        preprocess_integer_with_punctuation,
        preprocess_keep_letters_only,
    ]


def is_number_token(s: str) -> bool:
    """Returns True if s is a number token (integer or decimal)."""
    return should_normalize_integer_token(s) or split_decimal(s) is not None


def split_decimal(s: str) -> tuple[str, str, str] | None:
    """
    If s is a decimal number (e.g. "3.14" or "3,14"), return (integer_part, separator, decimal_part).
    Otherwise return None.
    """
    m = re.fullmatch(r"(\d+)([.,])(\d+)", s.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def should_normalize_integer_token(s: str) -> bool:
    """
    Returns True ONLY when:
    - The token contains an integer
    - It is NOT a decimal
    - Digits are NOT adjacent to letters
    - Digits are surrounded by non-letter characters (or boundaries)
    """
    s = s.strip()

    # Has at least one digit
    if not re.search(r"\d", s):
        return False

    # Is a decimal → do NOT normalize
    if re.fullmatch(r"\d+[.,]\d+", s):
        return False

    # Alphanumeric (letters touching digits) → do NOT normalize
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return False

    # Integer possibly wrapped by non-letter characters
    return bool(re.fullmatch(r"[^A-Za-z0-9]*\d+[^A-Za-z0-9]*", s))
