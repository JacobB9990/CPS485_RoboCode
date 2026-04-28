#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = ROOT / "logs" / "analysis"


def latest_summary_path() -> Path:
    candidates = sorted(ANALYSIS_DIR.glob("summary_*.json"))
    if not candidates:
        raise FileNotFoundError("No summary_*.json found under logs/analysis")
    return candidates[-1]


def main() -> int:
    summary_path = latest_summary_path()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    presentation_style = (ROOT / "Tex" / "robocode_ai_presentation.tex").read_text(encoding="utf-8")

    prompt = f"""Write a complete Beamer LaTeX slide deck for this Robocode Tank Royale project.

Use the same overall style, theme choices, author fields, and course framing as this reference file:
{presentation_style[:4000]}

The slide deck must cover:
- Experimental setup
- Per-bot architecture summary
- 1v1 results
- Melee results
- Map size ablation
- Learning curves
- Neuroevolution fitness progression
- Conclusions

Use the following structured experimental summary as ground truth. Embed these numeric findings into the slides and speaker-facing narrative:

{json.dumps(summary, indent=2)}

Requirements for the output .tex:
- Return only valid LaTeX source
- Use Beamer
- Include figures placeholders or \\includegraphics references for the plots in logs/analysis/figures
- Include concise interpretation on each results slide
- Keep terminology consistent with Tank Royale and Robocode
"""

    output_path = ANALYSIS_DIR / "llm_report_prompt.txt"
    output_path.write_text(prompt, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
