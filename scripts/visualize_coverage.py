#!/usr/bin/env python3
"""Visualize gloss coverage from a JSON file produced by text_to_gloss_to_pose --coverage-stats."""

import argparse
import json

# ANSI color codes
_RESET = "\033[0m"
_COLORS = {
    "lexicon":               "\033[92m",        # bright green (best / direct match)
    "decimal_parts":         "\033[96m",        # bright cyan (structured numeric decomposition)
    "language_backup":       "\033[93m",        # bright yellow (language rule fallback)
    "fingerspelling_backup": "\033[38;5;208m",  # orange (manual fallback)
    "placeholder":           "\033[95m",        # magenta (special placeholder token)
    "separator":             "\033[38;5;197m",  # same as unmatched — compound hyphen separator
    None:                    "\033[38;5;197m",  # strong magenta-red (not matched)
}

_LEGEND = [
    ("lexicon",               "matched via lexicon"),
    ("decimal_parts",         "matched via decimal decomposition"),
    ("language_backup",       "matched via language backup"),
    ("fingerspelling_backup", "matched via fingerspelling"),
    ("placeholder",           "matched via gloss placeholder"),
    (None,                    "not matched"),
]


def _colored(text: str, coverage_type) -> str:
    return f"{_COLORS[coverage_type]}{text}{_RESET}"


def _colored_token(token: dict) -> str:
    coverage_type = token.get("coverage_type")
    gloss = token["gloss"]
    if coverage_type == "decimal_parts":
        parts = token.get("fingerspelled_keys") or []
        if parts:
            return "".join(
                f"{_COLORS['decimal_parts'] if found is True else _COLORS.get(found, _COLORS[None])}{part}{_RESET}"
                for part, found in parts
            )
        return _colored(gloss, coverage_type)
    return _colored(gloss, coverage_type)


def _print_legend():
    print("Legend:")
    for coverage_type, label in _LEGEND:
        print(f"  {_colored('■', coverage_type)} {label}")
    print()


def visualize(coverage_path: str):
    with open(coverage_path, encoding="utf-8") as f:
        data = json.load(f)

    overall_coverage = data.get("coverage", 0.0)
    matched = data.get("matched_tokens", 0)
    total = data.get("total_tokens", 0)

    _print_legend()

    for sentence in data["sentences"]:
        sentence_text = sentence.get("text") or " ".join(
            t["word"] for t in sentence["tokens"] if t.get("word") and t.get("coverage_type") != "separator"
        )
        # Join tokens: attach separator tokens ("-") directly without spaces so that
        # "Online" + "-" + "Geldspiele" renders as "Online-Geldspiele", not "Online - Geldspiele".
        tokens = sentence["tokens"]
        colored_parts = []
        for i, t in enumerate(tokens):
            is_sep = t.get("coverage_type") == "separator"
            prev_is_sep = i > 0 and tokens[i - 1].get("coverage_type") == "separator"
            if i > 0 and not is_sep and not prev_is_sep:
                colored_parts.append(" ")
            colored_parts.append(_colored_token(t))
        colored_glosses = "".join(colored_parts)
        print(f"Sentence: {sentence_text}")
        print(f"Gloss:    {colored_glosses}")
        print()

    print(f"Overall coverage: {overall_coverage:.3f} ({matched}/{total} tokens matched)")


def main():
    parser = argparse.ArgumentParser(description="Visualize gloss coverage from a JSON coverage file.")
    parser.add_argument("coverage_json", help="Path to the coverage JSON file.")
    args = parser.parse_args()

    visualize(args.coverage_json)


if __name__ == "__main__":
    main()
