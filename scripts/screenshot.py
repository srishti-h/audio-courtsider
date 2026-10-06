"""Capture the dashboard for the README (needs `make serve` + `make web` running and Google Chrome).

uv run python scripts/screenshot.py
"""

import asyncio
from urllib.parse import quote

from playwright.async_api import async_playwright

GAME = "spain_laliga/2015-2016/2015-11-08 - 18-00 Barcelona 3 - 0 Villarreal"
URL = f"http://localhost:5173/?game={quote(GAME)}&start=5040&speed=5&feed_delay=8&threshold=0.2&autoplay=1"


async def main(wait_s: float = 13.5, out: str = "results/figures/dashboard.png") -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        await page.goto(URL)
        await page.wait_for_timeout(wait_s * 1000)
        await page.screenshot(path=out, full_page=True)
        await browser.close()
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
