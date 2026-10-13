"""Browser acceptance test against running Phase A API/frontend servers.

Run `sciforensics serve --config configs/demo.yaml` and `cd web && npm start`
first. Set DEMO_WEB_URL / DEMO_API_URL for another local deployment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, help="Write a README GIF of the verified flow")
    args = parser.parse_args()
    web = os.environ.get("DEMO_WEB_URL", "http://localhost:3000")
    root = Path(__file__).resolve().parents[1]
    frames: list[bytes] = []
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        failures: list[str] = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        requests: list[str] = []
        page.on(
            "request",
            lambda request: requests.append(request.url) if request.method == "POST" else None,
        )
        sockets: list[str] = []
        page.on("websocket", lambda socket: sockets.append(socket.url))
        page.goto(web)
        page.get_by_role("button", name="Run Rotated reuse example").wait_for()
        frames.append(page.screenshot())
        page.get_by_role("button", name="Run Rotated reuse example").click()
        page.get_by_role("link", name="Download PDF report").wait_for(timeout=120_000)
        assert page.locator(".match-lines line").count() > 0, "no SVG correspondences"
        assert page.get_by_text("verified", exact=True).count() > 0
        assert sockets, "WebSocket progress was never connected"
        frames.append(page.screenshot())
        page.get_by_role("button", name="Zoom in", exact=True).click()
        left = page.get_by_test_id("transform-left")
        right = page.get_by_test_id("transform-right")
        assert left.get_attribute("transform") == right.get_attribute("transform")
        pane = page.get_by_test_id("pane-left").bounding_box()
        assert pane
        before = left.get_attribute("transform")
        page.mouse.move(pane["x"] + 100, pane["y"] + 100)
        page.mouse.down()
        page.mouse.move(pane["x"] + 140, pane["y"] + 120, steps=4)
        page.mouse.up()
        assert left.get_attribute("transform") != before
        assert left.get_attribute("transform") == right.get_attribute("transform")
        page.get_by_role("button", name="Fit images").click()
        page.get_by_role("button", name="Verified regions", exact=True).click()
        assert page.locator(".image-pane polygon").count() == 2
        frames.append(page.screenshot())
        page.get_by_role("button", name="Keypoints", exact=True).click()
        assert "keypoints_left" in (
            page.locator(".image-pane image").first.get_attribute("href") or ""
        )
        page.get_by_role("button", name="Heatmaps", exact=True).click()
        frames.append(page.screenshot())
        posts = len(requests)
        score = page.locator(".score").inner_text()
        page.locator("#match-threshold").evaluate(
            "el => { el.value = '3'; el.dispatchEvent(new Event('input', {bubbles: true})); }"
        )
        # React listens to native input updates; keyboard interaction exercises the actual control.
        page.locator("#match-threshold").focus()
        page.keyboard.press("End")
        assert page.locator(".score").inner_text() != score, "threshold did not re-score"
        assert len(requests) == posts, "slider reran analysis"
        frames.append(page.screenshot())
        with page.expect_download() as download:
            page.get_by_role("link", name="Download PDF report").click()
        pdf = Path(download.value.path())
        assert pdf.read_bytes().startswith(b"%PDF"), "download was not a PDF"
        import pymupdf

        with pymupdf.open(pdf) as doc:
            text = " ".join("".join(p.get_text() for p in doc).split())
            assert "absence of a finding is not proof of originality" in text.lower()
            assert "small reused regions." in text
            assert "weights sha256" in text.lower()
        page.get_by_role("button", name="Run Unrelated cell panels example").click()
        page.get_by_role("link", name="Download PDF report").wait_for(state="hidden")
        page.get_by_role("link", name="Download PDF report").wait_for(timeout=120_000)
        assert page.get_by_text("No evidence of reuse found", exact=True).count() > 0
        assert page.locator(".badge").filter(has_text="refused").count() > 0
        frames.append(page.screenshot())
        page.get_by_role("button", name="Run Cloned cell region example").click()
        page.get_by_role("link", name="Download PDF report").wait_for(state="hidden")
        page.get_by_role("link", name="Download PDF report").wait_for(timeout=120_000)
        import re

        page.get_by_role("heading", name=re.compile(r"Cloned regions\s*\(\s*1")).wait_for()
        page.get_by_role("button", name="Clone mask", exact=True).click()
        assert "copymove" in (page.locator(".image-pane image").first.get_attribute("href") or "")
        frames.append(page.screenshot())
        page.get_by_role("button", name="Compare pair", exact=True).click()
        inputs = page.locator('input[type="file"]')
        inputs.nth(0).set_input_files(root / "inputs/base_cells_2.png")
        inputs.nth(1).set_input_files(root / "inputs/base_cells_2.png")
        page.get_by_role("button", name="Run analysis", exact=True).click()
        page.get_by_role("link", name="Download PDF report").wait_for(timeout=120_000)
        assert page.get_by_text("verified", exact=True).count() > 0
        assert not failures, failures
        browser.close()
    if args.record:
        import io

        from PIL import Image

        images = [Image.open(io.BytesIO(frame)).resize((960, 700)) for frame in frames]
        args.record.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(args.record, save_all=True, append_images=images[1:], duration=1600, loop=0)
    print(
        "PASS: examples, uploads, synced pan/zoom, SVG lines, overlays, cached re-scoring, "
        "WebSocket progress, PDF audit/disclaimer, negative control and CMFD"
    )


if __name__ == "__main__":
    main()
