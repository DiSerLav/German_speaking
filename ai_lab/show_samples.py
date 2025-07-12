import json
import random
from pathlib import Path
from collections import defaultdict


def show_samples(path_in: str | Path, n: int = 3):
    """Выводит *n* случайных фраз из каждого топика.

    Args:
        path_in: путь к predicted.jsonl, содержащему поля "topic" и "text".
        n: количество примеров на тему.
    """
    path_in = Path(path_in)
    buckets: dict[int, list[str]] = defaultdict(list)

    with path_in.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            topic = obj.get("topic", -1)
            text = obj.get("text", "")
            buckets[topic].append(text)

    for topic in sorted(buckets):
        items = buckets[topic]
        sample = items if len(items) <= n else random.sample(items, n)
        print(f"\n=== Topic {topic} (docs: {len(items)}) ===")
        for i, txt in enumerate(sample, 1):
            short = txt if len(txt) <= 300 else txt[:297] + "…"
            print(f" {i}. {short}")


if __name__ == "__main__":
    CUR_DIR = Path(__file__).parent
    show_samples(CUR_DIR / "data" / "predicted.jsonl", 4)