"""Record the README GIF from the static demo (needs `npm run build:pages` + `vite preview` on :4173).

uv run python scripts/record_gif.py
"""

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://localhost:4173/audio-courtsider/?match={m}&goal=2&fd=8&speed=4"
MATCH = "2015-11-08-18-00-barcelona-3-0-villarreal"
OUT = Path("results/figures/demo.gif")


async def record(seconds: float, workdir: Path) -> float:
    """Grab frames by screenshotting in a loop (no extra browser downloads); returns the achieved fps."""
    import time

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome")
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(URL.format(m=MATCH))
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            await page.screenshot(path=str(workdir / f"f{n:05d}.png"))
            n += 1
        fps = n / (time.perf_counter() - t0)
        await browser.close()
    return fps


def main(start: float = 4.0, seconds: float = 16.0) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fps = asyncio.run(record(start + seconds, Path(tmp)))
        video = Path(tmp) / "f%05d.png"
        palette = Path(tmp) / "palette.png"
        filt = "fps=10,scale=1100:-1:flags=lanczos"
        src = ["-framerate", f"{fps:.3f}", "-ss", str(start), "-t", str(seconds), "-i", str(video)]
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", *src, "-vf", f"{filt},palettegen=max_colors=128", str(palette)],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                *src,
                "-i",
                str(palette),
                "-lavfi",
                f"{filt} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4",
                str(OUT),
            ],
            check=True,
        )
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    assert shutil.which("ffmpeg"), "ffmpeg required"
    main()
