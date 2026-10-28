"""Regenerate every figure and terminal capture used in the README from real runs.

Usage (from the repo root, with the dev extras installed):
    python scripts/build_docs_assets.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"

TERM_BG, TERM_FG, TERM_DIM = "#1a1a19", "#f0efec", "#898781"


def render_terminal(text: str, title: str, path: Path, max_lines: int = 60) -> None:
    """Render captured plain-text output as a terminal-style PNG."""
    lines = text.rstrip().splitlines()
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], "..."]
    width = max(len(line) for line in lines)
    fig_w = min(max(width * 0.092 + 0.8, 7), 15)
    fig_h = len(lines) * 0.152 + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=TERM_BG)
    fig.text(
        0.02, 1 - 0.35 / fig_h, title, color=TERM_DIM, family="monospace", fontsize=9, va="top"
    )
    for i, color in enumerate(("#e34948", "#eda100", "#1baf7a")):
        fig.patches.append(
            plt.Circle((fig_w - 0.55 + i * 0.18, fig_h - 0.28), 0.055, color=color,
                       transform=fig.dpi_scale_trans, figure=fig)
        )  # fmt: skip
    fig.text(
        0.02,
        1 - 0.7 / fig_h,
        "\n".join(lines),
        color=TERM_FG,
        family="monospace",
        fontsize=8.6,
        va="top",
        linespacing=1.25,
    )
    fig.savefig(path, dpi=160, facecolor=TERM_BG)
    plt.close(fig)


def run(cmd: list[str]) -> str:
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return done.stdout + done.stderr


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    for name, cfg in (("spy", "configs/spy.yaml"),):
        out = ROOT / "reports" / name
        text = run([py, "-m", "quantvol.cli", "run", "-c", cfg, "-o", str(out)])
        (out / "cli_output.txt").write_text(text, encoding="utf-8")
        for fig in (
            "performance",
            "stress",
            "parameter_sweep",
            "regime_attribution",
            "walk_forward",
        ):
            src = out / f"{fig}.png"
            if src.exists():
                (IMAGES / f"{name}_{fig}.png").write_bytes(src.read_bytes())
        if name == "spy":
            head = text.split("Regime attribution")[0].rstrip()
            render_terminal(
                head, "$ quantvol run --config configs/spy.yaml", IMAGES / "cli_run.png"
            )

    tests = run([py, "-m", "pytest", "-q", "--cov", "--color=no", "-p", "no:cacheprovider"])
    keep = [ln for ln in tests.splitlines() if ln.strip() and not ln.startswith("Required")]
    render_terminal(
        "\n".join(keep),
        "$ pytest --cov",
        IMAGES / "tests_coverage.png",
        max_lines=40,
    )
    (ROOT / "reports" / "pytest_output.txt").write_text(tests, encoding="utf-8")
    print("assets written to", IMAGES)


if __name__ == "__main__":
    main()
