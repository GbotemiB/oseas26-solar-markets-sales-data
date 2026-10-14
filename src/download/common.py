"""Shared helper for the download scripts."""
import json
import time
from datetime import date

import requests


def stream_to_file(url: str, out_path, meta: dict, retries: int = 30, timeout: int = 60) -> None:
    """Download a large file in chunks and write a .json next to it (source, licence, date).

    Skips the download if the file is already there. Writes to a .part file first, so a broken
    download never looks like a finished one. If the connection drops, it resumes from the
    .part file (HTTP Range request), up to `retries` times. Running it again also resumes.
    """
    if out_path.exists():
        print(f"  {out_path.name} already there, skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_name(out_path.name + ".part")

    total = None
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416 and done:        # asked for bytes past the end: already complete
                    break
                r.raise_for_status()
                if done and r.status_code != 206:        # server ignored Range: start again
                    done = 0
                total = done + int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
                with open(part, "ab" if done else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if total is None or part.stat().st_size == total:
                break
            print(f"  incomplete ({part.stat().st_size} of {total} bytes), resuming")
        except (requests.ConnectionError, requests.Timeout) as err:
            print(f"  attempt {attempt}/{retries} interrupted at {part.stat().st_size / 1e6:.0f} MB "
                  f"({type(err).__name__}), resuming")
            time.sleep(min(2 * attempt, 30))
    else:
        raise IOError(f"Download of {url} did not finish after {retries} attempts. "
                      f"Run the same command again to resume from {part.name}.")

    size = part.stat().st_size
    part.rename(out_path)
    meta = {**meta, "url": url, "downloaded": date.today().isoformat()}
    out_path.with_name(out_path.name + ".json").write_text(json.dumps(meta, indent=2))
    print(f"  {out_path.name}: {size / 1e6:.1f} MB")
