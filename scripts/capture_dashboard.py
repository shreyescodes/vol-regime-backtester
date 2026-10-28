"""Launch the Streamlit dashboard headless and screenshot each tab (needs `quantvol[docs]`).

Usage:
    python scripts/capture_dashboard.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
DASHBOARD = ROOT / "src" / "quantvol" / "dashboard.py"
TABS = ["Overview", "Equity & drawdown", "Regimes", "Stress tests", "Robustness"]
SLUGS = ["overview", "equity", "regimes", "stress", "robustness"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    port = free_port()
    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
            "--server.headless=true", f"--server.port={port}",
            "--browser.gatherUsageStats=false",
            "--theme.base=light", "--client.toolbarMode=minimal",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5
            )
            for _ in range(60):
                try:
                    page.goto(f"http://127.0.0.1:{port}", timeout=3000)
                    break
                except Exception:
                    time.sleep(1)
            page.wait_for_selector("text=QuantVol: regime-aware", timeout=60000)
            page.wait_for_selector("[data-testid=stTab]", timeout=120000)
            for tab, slug in zip(TABS, SLUGS, strict=True):
                page.get_by_role("tab", name=tab).click()
                page.wait_for_timeout(2500)
                assert page.locator("[data-testid=stException]").count() == 0, f"error on {tab}"
                page.screenshot(path=str(IMAGES / f"dashboard_{slug}.png"), full_page=True)
                print("captured", slug)
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
