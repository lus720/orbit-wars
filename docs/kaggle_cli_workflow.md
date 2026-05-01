# Orbit Wars Kaggle CLI Workflow

Useful commands for submitting `submission.py`, downloading episode replays,
downloading your own agent logs, and pulling public replays from leaderboard
submissions.

## Setup

Install or upgrade the official Kaggle CLI. The episode/replay/log commands are
available in Kaggle CLI 2.0.2+.

```bash
conda activate kaggle
python -m pip install -U "kaggle>=2.0.2"
kaggle --version
```

## Submit

```bash
kaggle competitions submit orbit-wars \
  -f submission.py \
  -m "current submission"
```

Check your submissions.

```bash
kaggle competitions submissions orbit-wars
kaggle competitions submissions orbit-wars -v > output/orbit-wars-submissions.csv
```

Some CLI versions do not print the numeric submission id in
`competitions submissions`. If that happens, open the submission in the Kaggle UI
or use the leaderboard/internal API commands below to get `submissionId`.

## Your Replays And Logs

List episodes played by a submission.

```bash
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions episodes <SUBMISSION_ID> -v > output/submission-<SUBMISSION_ID>-episodes.csv
```

Download an episode replay.

```bash
mkdir -p replay/kaggle
kaggle competitions replay <EPISODE_ID> -p replay/kaggle
```

This writes:

```text
replay/kaggle/episode-<EPISODE_ID>-replay.json
```

If the official CLI fails with `KeyError: 'content-length'`, use the local SDK
helper instead. It uses the same Kaggle authentication but writes the response
body directly.

```bash
conda run -n kaggle python tools/download_kaggle_replay.py <EPISODE_ID> \
  -p replay/kaggle
```

Download logs for your own agent in that episode. `AGENT_INDEX` is the 0-based
slot in the episode.

```bash
kaggle competitions logs <EPISODE_ID> <AGENT_INDEX> -p replay/kaggle
```

This writes:

```text
replay/kaggle/episode-<EPISODE_ID>-agent-<AGENT_INDEX>-logs.json
```

Logs are private to your own agents. Public replays are okay to download for
other teams, but their logs are not available.

Analyze downloaded files with the local tooling:

```bash
python analyze_replays.py --replay-dir replay/kaggle
```

## Top Leaderboard Replays

Leaderboard rows expose `submissionId` for each ranked team. Once you have a
top player's `submissionId`, list that submission's public episodes, then
download selected `episodeId` values.

```bash
TOP_SUBMISSION_ID=<SUBMISSION_ID>
mkdir -p replay/kaggle-top-${TOP_SUBMISSION_ID}

kaggle competitions episodes "$TOP_SUBMISSION_ID" -v \
  > "replay/kaggle-top-${TOP_SUBMISSION_ID}/episodes.csv"

kaggle competitions replay <EPISODE_ID> \
  -p "replay/kaggle-top-${TOP_SUBMISSION_ID}"
```

In practice, current CLI versions may return `403` for
`competitions episodes` on other teams' public submissions, and
`competitions replay` may fail with `KeyError: 'content-length'`. If either
happens, use `EpisodeService/ListEpisodes` below to list episode ids and
`tools/download_kaggle_replay.py` to download the replay.

## Leaderboard Submission IDs

If the web UI is inconvenient, this uses Kaggle's current internal API to print
top leaderboard `submissionId` values. It starts with a normal browser-like GET
to receive the XSRF cookie, then calls `LeaderboardService/GetLeaderboard`.

```bash
mkdir -p /tmp/kaggle-orbit
curl -L -s -c /tmp/kaggle-orbit/cookies.txt \
  "https://www.kaggle.com/competitions/orbit-wars/leaderboard" >/dev/null

XSRF=$(
  awk '$6=="XSRF-TOKEN"{print $7}' /tmp/kaggle-orbit/cookies.txt
)

curl -L -s -b /tmp/kaggle-orbit/cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-XSRF-TOKEN: ${XSRF}" \
  -X POST \
  "https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard" \
  --data '{"competitionId":138420}' \
  > /tmp/kaggle-orbit/leaderboard.json
```

Print the top 20 rows as `rank,score,teamId,submissionId,teamName`.

```bash
python - <<'PY'
import csv
import json
import sys

with open("/tmp/kaggle-orbit/leaderboard.json", encoding="utf-8") as f:
    data = json.load(f)

teams = {team["teamId"]: team.get("teamName", "") for team in data.get("teams", [])}
writer = csv.writer(sys.stdout)
writer.writerow(["rank", "score", "teamId", "submissionId", "teamName"])
for row in data.get("publicLeaderboard", [])[:20]:
    writer.writerow([
        row.get("rank"),
        row.get("displayScore"),
        row.get("teamId"),
        row.get("submissionId"),
        teams.get(row.get("teamId"), ""),
    ])
PY
```

Then use the chosen `submissionId`:

```bash
kaggle competitions episodes <LEADERBOARD_SUBMISSION_ID> -v \
  > replay/top-submission-episodes.csv
kaggle competitions replay <EPISODE_ID> -p replay/kaggle
```

## Direct Episode List API Fallback

The internal `EpisodeService/ListEpisodes` endpoint can list public episodes
for any known `submissionId`. It also includes each agent's 0-based slot
`index`; if `index` is missing, treat it as slot `0`.

```bash
SUBMISSION_ID=<SUBMISSION_ID>

curl -L -s -b /tmp/kaggle-orbit/cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-XSRF-TOKEN: ${XSRF}" \
  -X POST \
  "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes" \
  --data "{\"submissionId\":${SUBMISSION_ID}}" \
  > "/tmp/kaggle-orbit/submission-${SUBMISSION_ID}-episodes.json"
```

Print a compact episode table:

```bash
SUBMISSION_ID=<SUBMISSION_ID> python - <<'PY'
import csv
import json
import os
import sys

submission_id = int(os.environ["SUBMISSION_ID"])
path = f"/tmp/kaggle-orbit/submission-{submission_id}-episodes.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

writer = csv.writer(sys.stdout)
writer.writerow(["episodeId", "state", "created", "agentIndex", "reward", "opponentSubmissionIds"])
for episode in data.get("episodes", [])[:50]:
    agents = episode.get("agents", [])
    mine = next((agent for agent in agents if agent.get("submissionId") == submission_id), {})
    opponents = [str(agent.get("submissionId")) for agent in agents if agent.get("submissionId") != submission_id]
    writer.writerow([
        episode.get("id"),
        episode.get("state"),
        episode.get("createTime"),
        mine.get("index", 0),
        mine.get("reward"),
        " ".join(opponents),
    ])
PY
```

## Direct Replay API Fallback

Prefer the CLI, but the current Kaggle frontend also has an internal replay
endpoint. It returns the replay payload for public episodes when the XSRF cookie
is present.

```bash
EPISODE_ID=<EPISODE_ID>

curl -L -s -b /tmp/kaggle-orbit/cookies.txt \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-XSRF-TOKEN: ${XSRF}" \
  -X POST \
  "https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay" \
  --data "{\"episodeId\":${EPISODE_ID}}" \
  > "replay/kaggle/episode-${EPISODE_ID}-replay.json"
```
