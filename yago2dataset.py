import argparse, os, json, re
from datetime import datetime
import pathlib as pl
from typing import Tuple

Fact = Tuple[str, str, str, str]


def load_yago(path: pl.Path, relations: set[str]) -> set[Fact]:

    print("loading YAGO facts...", end="")
    facts = {}  # { (subj, rel, obj) => ts }
    with open(path / "yago-facts.ttl") as f:
        i = 0
        for line in f:
            if i % 1000000 == 0:
                print(".", end="", flush=True)
            i += 1
            try:
                subj, rel, obj, _ = line.split("\t")
            except ValueError:
                continue
            if not rel in relations:
                continue
            facts[(subj, rel, obj)] = ""
    print("done!")

    print("loading YAGO metadata...", end="")
    with open(path / "yago-meta-facts.ntx") as f:
        i = 0
        for line in f:
            if i % 1000000 == 0:
                print(".", end="", flush=True)
            i += 1
            try:
                _, subj, rel, obj, _, metakey, metaval = line.split("\t")
            except ValueError:
                continue
            if not (subj, rel, obj) in facts:
                continue
            m = re.match(
                r"\"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+)Z\"\^\^xsd:dateTime",
                '"1997-01-01T00:00:00Z"^^xsd:dateTime',
            )
            assert not m is None
            metaval = m.group(1)
            if metakey == "schema:startDate":
                if facts[(subj, rel, obj)] == "":
                    facts[(subj, rel, obj)] = f"{metaval}--"
                else:
                    facts[(subj, rel, obj)] = f"{metaval}{facts[(subj, rel, obj)]}"
            elif metakey == "schema:endDate":
                if facts[(subj, rel, obj)] == "":
                    facts[(subj, rel, obj)] = f"--{metaval}"
                else:
                    facts[(subj, rel, obj)] = f"{facts[(subj, rel, obj)]}{metaval}"
    print("done!")

    ts_facts = {key + (ts,) for key, ts in facts.items() if ts != ""}
    print(f"NOTE: dropping {len(facts) - len(ts_facts)} facts without timesamps.")
    return ts_facts


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", "-i", type=pl.Path)
parser.add_argument("--relations", "-r", default=set(), nargs="*")
parser.add_argument("--output-dir", "-o", type=pl.Path)
args = parser.parse_args()

relations = set(args.relations)
facts = load_yago(args.input_dir, relations)
print(f"found {len(facts)} facts for {len(relations)} relations.")
entities = {f[0] for f in facts} | {f[2] for f in facts}
print(f"found {len(entities)} unique entities.")
timestamps = {f[3] for f in facts}
print(f"found {len(timestamps)} unique timestamps.")


os.makedirs(args.output_dir, exist_ok=True)


print(f"writing entity2id.json to {args.output_dir}...", end="")
with open(args.output_dir / "entity2id.json", "w") as f:
    json.dump({entity: i for i, entity in enumerate(entities)}, f)
print("done!")

print(f"writing relation2id.json to {args.output_dir}...", end="")
with open(args.output_dir / "relation2id.json", "w") as f:
    json.dump({rel: i for i, rel in enumerate(relations)}, f)
print("done!")

print(f"writing ts2id.json to {args.output_dir}...", end="")
with open(args.output_dir / "ts2id.json", "w") as f:
    json.dump({ts: i for i, ts in enumerate(timestamps)}, f)
print("done!")


def latest_datetime(ts: str) -> datetime:
    """
    :param ts: timestamp with a format of either 'START--', '--END' or
        'START--END', where START and END being in an ISO format.
    """
    start, end = ts.split("--")
    if end == "":
        return datetime.fromisoformat(start)
    return datetime.fromisoformat(end)


facts = sorted(facts, key=lambda fact: latest_datetime(fact[3]))  # type: ignore

print(f"writing train.txt to {args.output_dir}...", end="")
train = facts[: int(0.8 * len(facts))]
with open(args.output_dir / "train.txt", "w") as f:
    for subj, rel, obj, ts in train:
        f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
print("done!")

print(f"writing valid.txt to {args.output_dir}...", end="")
valid = facts[int(0.8 * len(facts)) : int(0.9 * len(facts))]
with open(args.output_dir / "valid.txt", "w") as f:
    for subj, rel, obj, ts in valid:
        f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
print("done!")

print(f"writing test.txt to {args.output_dir}...", end="")
test = facts[int(0.9 * len(facts)) :]
with open(args.output_dir / "test.txt", "w") as f:
    for subj, rel, obj, ts in test:
        f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
print("done!")
