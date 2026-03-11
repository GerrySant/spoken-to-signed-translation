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
    Strip surrounding non-letter punctuation from standalone integer tokens.

    This handles glosses like "800." or "(5)" where the lexicon stores the bare integer.
    Decimals (e.g. "3.14") and alphanumeric tokens (e.g. "A3") are left untouched because:
    - Decimals are handled separately via split_decimal().
    - Alphanumeric tokens are identifiers, not numbers (e.g. "B2", "3D").
    """
    s = s.strip()

    # Keep decimals untouched — they are decomposed separately by split_decimal()
    if re.fullmatch(r"\d+[.,]\d+", s):
        return s

    # Reject alphanumeric tokens (letters touching digits are identifiers, not numbers)
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return s

    # Strip any surrounding non-alphanumeric characters (e.g. trailing dot, parentheses)
    m = re.fullmatch(r"[^A-Za-z0-9]*([0-9]+)[^A-Za-z0-9]*", s)
    if m:
        return m.group(1)

    return s


# Languages that use the apostrophe as a thousands separator.
# In Swiss German (de), French (fr), and Italian (it), large numbers are written
# as e.g. "1'400" or "1'400'000". This is unambiguous (unlike dot/comma, which
# are also used as decimal separators) so we can safely strip the apostrophes.
_APOSTROPHE_THOUSANDS_LANGUAGES = {"de", "fr", "it"}


def preprocess_thousands_separator(s: str) -> str:
    """
    Remove apostrophe thousands separators, preserving any surrounding punctuation.

    Examples: "1'400" → "1400", "1'400." → "1400.", "1'400'000" → "1400000"

    Surrounding non-alphanumeric characters (e.g. a trailing dot) are preserved so
    that preprocess_integer_with_punctuation can strip them in the next step.
    Only applied for languages where apostrophe is the standard thousands separator
    (see _APOSTROPHE_THOUSANDS_LANGUAGES).
    """
    s = s.strip()
    # Allow optional surrounding non-alphanumeric punctuation (e.g. trailing dot)
    m = re.fullmatch(r"([^A-Za-z0-9]*)(\d{1,3}(?:'\d{3})+)([^A-Za-z0-9]*)", s)
    if m:
        return m.group(1) + m.group(2).replace("'", "") + m.group(3)
    return s


def preprocess_leading_zeros(s: str) -> str:
    """
    Strip leading zeros from pure integer strings (e.g. "07" → "7").

    Some glossers or source texts produce zero-padded numbers (e.g. dates formatted
    as "07"). The lexicon typically stores the canonical form without leading zeros.
    Only applied to pure digit strings to avoid altering e.g. "007" in a code context.
    """
    if re.fullmatch(r"0\d+", s):
        return str(int(s))
    return s


def preprocess_keep_letters_only(s: str) -> str:
    """Keep only Unicode letters."""
    return "".join(ch for ch in s if ch.isalpha())


def get_progressive_gloss_normalizers(spoken_language: str = None) -> List[Callable[[str], str]]:
    """
    Return the ordered list of progressive gloss normalization steps.

    Each step is applied cumulatively during lookup: if a gloss is not found after
    step N, step N+1 is applied and lookup is retried. The first entry (None)
    represents the original gloss with no transformation.

    The thousands-separator step is inserted only for languages that use the
    apostrophe as a thousands separator (de, fr, it), because dot/comma separators
    are ambiguous with decimal notation and cannot be stripped safely.
    """
    steps = [
        None,  # original gloss — try an exact match first
        preprocess_lower_strip,
        preprocess_rule_based,
        preprocess_spacylemma,
    ]
    if spoken_language in _APOSTROPHE_THOUSANDS_LANGUAGES:
        # "1'400" → "1400": safe only for languages where apostrophe is unambiguously
        # a thousands separator, never a decimal separator.
        steps.append(preprocess_thousands_separator)
    steps += [
        preprocess_integer_with_punctuation,  # "800." → "800"
        preprocess_leading_zeros,             # "07"   → "7"
        preprocess_keep_letters_only,
    ]
    return steps


def is_apostrophe_thousands(s: str) -> bool:
    """
    Return True if s is an integer written with apostrophe thousands separators.

    Examples: "1'400" → True, "1'400'000" → True, "1400" → False, "3.14" → False.

    Used by is_number_token() to extend number detection to Swiss-style notation.
    Note: surrounding punctuation (e.g. trailing dot) is NOT handled here — callers
    that need to tolerate it should strip it before calling this function.
    """
    return bool(re.fullmatch(r"\d{1,3}(?:'\d{3})+", s.strip()))


def is_number_token(s: str) -> bool:
    """
    Return True if s represents a number in any of the forms we handle.

    Recognised forms:
    - Standalone integer, possibly with surrounding punctuation: "42", "800.", "(5)"
    - Decimal number: "3.14", "3,14"
    - Apostrophe-thousands integer: "1'400", "1'400'000"

    Used in lookup() to decide whether to apply the number_placeholder fallback:
    only number-like glosses should fall back to a generic number sign, not arbitrary
    unmatched glosses.
    """
    return should_normalize_integer_token(s) or split_decimal(s) is not None or is_apostrophe_thousands(s)


def split_decimal(s: str) -> tuple[str, str, str] | None:
    """
    If s is a decimal number, return (integer_part, separator, decimal_part).
    Otherwise return None.

    Examples: "3.14" → ("3", ".", "14"), "3,14" → ("3", ",", "14"), "314" → None.

    Used by lookup() to decompose decimal glosses into independently signable parts.
    The separator itself (dot or comma) is looked up as a separate sign; if not found,
    it is skipped rather than replaced by the number_placeholder, because a separator
    is not a number.
    """
    m = re.fullmatch(r"(\d+)([.,])(\d+)", s.strip())
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def should_normalize_integer_token(s: str) -> bool:
    """
    Return True only when s contains a standalone integer that can safely be normalized.

    A token qualifies when:
    - It contains at least one digit.
    - It is NOT a decimal number (those are handled by split_decimal).
    - No digit is directly adjacent to a letter (rules out identifiers like "A3", "3D").
    - The digits are surrounded only by non-letter characters or string boundaries
      (e.g. "800.", "(5)" qualify; "1'400" does not — the apostrophe sits between digits).

    Used to guard the word-normalization step in lookup() so that only integer-type
    glosses trigger integer-specific fallbacks (e.g. leading-zero stripping on the word).
    """
    s = s.strip()

    # Must contain at least one digit
    if not re.search(r"\d", s):
        return False

    # Decimals are handled separately — do not normalise them here
    if re.fullmatch(r"\d+[.,]\d+", s):
        return False

    # Alphanumeric tokens (letters adjacent to digits) are identifiers, not numbers
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return False

    # Accept integers optionally wrapped by non-letter, non-digit characters
    return bool(re.fullmatch(r"[^A-Za-z0-9]*\d+[^A-Za-z0-9]*", s))
