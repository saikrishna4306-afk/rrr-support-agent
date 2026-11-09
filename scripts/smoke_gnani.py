"""Private Gnani smoke test; requires GNANI_API_KEY and a local WAV path.

Usage: python scripts/smoke_gnani.py path\\to\\recording.wav
"""
import asyncio
import sys
from pathlib import Path

from backend.providers.gnani import GnaniSTTProvider


async def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke_gnani.py path\\to\\recording.wav")
    result = await GnaniSTTProvider().transcribe(Path(sys.argv[1]).read_bytes())
    print(f"[{result.language or 'unknown'}] {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
