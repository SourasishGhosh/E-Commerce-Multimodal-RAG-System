import json
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path("data/feedback_log.jsonl")
SCORE_FILE = Path("data/rerank_scores.json")

def compute_rerank_scores():
    click_counts: dict[str, int] = defaultdict(int)

    with open(LOG_FILE) as f:
        for line in f:
            event = json.loads(line)
            click_counts[event["product_id"]] += 1

    # Normalize to a [0, 1] popularity boost factor
    max_clicks = max(click_counts.values(), default=1)
    scores = {pid: count / max_clicks for pid, count in click_counts.items()}

    with open(SCORE_FILE, "w") as f:
        json.dump(scores, f)

    print(f"Rerank scores written for {len(scores)} products.")

if __name__ == "__main__":
    compute_rerank_scores()