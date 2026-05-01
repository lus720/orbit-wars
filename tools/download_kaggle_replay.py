#!/usr/bin/env python3
"""Download a Kaggle simulation replay without requiring Content-Length.

Kaggle CLI 2.1.0 can fail on Orbit Wars replays with:
KeyError: 'content-length'. This helper uses the same authenticated Kaggle SDK
endpoint but writes the response body directly.
"""

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetEpisodeReplayRequest, KaggleApi


def parse_args():
    parser = argparse.ArgumentParser(description="Download an Orbit Wars episode replay JSON.")
    parser.add_argument("episode_id", type=int, help="Kaggle episode id.")
    parser.add_argument(
        "-p",
        "--path",
        default="replay/kaggle",
        help="Output directory. Defaults to replay/kaggle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode-{args.episode_id}-replay.json"

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiGetEpisodeReplayRequest()
        request.episode_id = args.episode_id
        response = kaggle.competitions.competition_api_client.get_episode_replay(request)
        response.raise_for_status()
        payload = response.content

    if not payload:
        raise RuntimeError(f"Episode {args.episode_id} replay response was empty")

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Episode {args.episode_id} replay response was not valid JSON") from exc

    out_path.write_bytes(payload)
    print(
        f"Replay downloaded to: {out_path} "
        f"({out_path.stat().st_size} bytes, steps={len(parsed.get('steps', []))})"
    )


if __name__ == "__main__":
    main()
