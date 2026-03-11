#!/usr/bin/env python3
"""Export gloss coverage from a JSON file to an HTML report."""

import argparse
import html
import json

_COLORS = {
    "lexicon":               "#00e676",  # bright green (best / direct match)
    "decimal_parts":         "#00e5ff",  # bright cyan (structured numeric decomposition)
    "language_backup":       "#ffee58",  # bright yellow (language rule fallback)
    "fingerspelling_backup": "#ff9100",  # orange (manual fallback)
    "placeholder":           "#ea80fc",  # magenta (special placeholder token)
    None:                    "#ff1744",  # strong magenta-red (not matched)
}

_LEGEND = [
    ("lexicon",               "matched via lexicon"),
    ("decimal_parts",         "matched via decimal decomposition"),
    ("language_backup",       "matched via language backup"),
    ("fingerspelling_backup", "matched via fingerspelling"),
    ("placeholder",           "matched via gloss placeholder"),
    (None,                    "not matched"),
]


def _span(text: str, coverage_type) -> str:
    color = _COLORS[coverage_type]
    return f'<span style="color:{color}" title="{coverage_type or "not matched"}">{html.escape(text)}</span>'


def _token_html(token: dict) -> str:
    coverage_type = token.get("coverage_type")
    gloss = token["gloss"]
    if coverage_type == "decimal_parts":
        parts = token.get("fingerspelled_keys") or []
        if parts:
            return "".join(
                _span(part, "decimal_parts" if found is True else (found if found in _COLORS else None))
                for part, found in parts
            )
    return _span(gloss, coverage_type)


def _legend_html() -> str:
    items = " &nbsp;&nbsp; ".join(
        f'<span style="color:{_COLORS[ct]}">&#9632;</span> {label}'
        for ct, label in _LEGEND
    )
    return f'<p style="font-family:monospace;font-size:0.9em">{items}</p>'


def build_html(data: dict) -> str:
    overall = data.get("coverage", 0.0)
    matched = data.get("matched_tokens", 0)
    total = data.get("total_tokens", 0)

    rows = []
    for sentence in data["sentences"]:
        sentence_text = sentence.get("text") or " ".join(
            t["word"] for t in sentence["tokens"] if t.get("word")
        )
        gloss_html = " ".join(_token_html(t) for t in sentence["tokens"])
        rows.append(
            f"""
            <tr>
              <td style="padding:4px 8px;color:#ccc">{html.escape(sentence_text or "")}</td>
              <td style="padding:4px 8px;font-family:monospace">{gloss_html}</td>
            </tr>"""
        )

    legend = _legend_html()
    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gloss Coverage Report</title>
  <style>
    body  {{ background:#1e1e1e; color:#e0e0e0; font-family:sans-serif; padding:2em; }}
    h1    {{ font-size:1.4em; margin-bottom:0.2em; }}
    table {{ border-collapse:collapse; width:100%; margin-top:1em; }}
    th    {{ text-align:left; padding:6px 8px; border-bottom:1px solid #444; color:#aaa; font-weight:normal; }}
    tr:nth-child(even) {{ background:#252525; }}
    .summary {{ margin-top:1.5em; font-size:0.95em; color:#bbb; }}
  </style>
</head>
<body>
  <h1>Gloss Coverage Report</h1>
  {legend}
  <table>
    <thead><tr><th>Sentence</th><th>Gloss</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <p class="summary">Overall coverage: <strong>{overall:.3f}</strong> ({matched}/{total} tokens matched)</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Export gloss coverage to an HTML report.")
    parser.add_argument("--coverage-json", required=True, help="Path to the coverage JSON file.")
    parser.add_argument("--output-html", required=True, help="Path for the output HTML file.")
    args = parser.parse_args()

    with open(args.coverage_json, encoding="utf-8") as f:
        data = json.load(f)

    report = build_html(data)

    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {args.output_html}")



if __name__ == "__main__":
    main()
