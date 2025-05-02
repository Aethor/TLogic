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
                '"1997-01-01T00:00:00Z"^^xsd:dateTime'
            )
            assert not m is None
            metaval = datetime.fromisoformat(m.group(1))
            if metakey == "startDate":
                facts[(subj, rel, obj)] = f"{metaval}:{facts[(subj, rel, obj)]}"
            elif metakey == "endDate":
                facts[(subj, rel, obj)] = f"{facts[(subj, rel, obj)]}:{metaval}"
    print("done!")

    return {key + (ts,) for key, ts in facts.items()}


parser = argparse.ArgumentParser()
parser.add_argument("--input-file", "-i", type=pl.Path)
parser.add_argument("--relations", "-r", default=set(), nargs="*")
parser.add_argument("--output-dir", "-o", type=pl.Path)
args = parser.parse_args()

relations = set(args.relations)
facts = load_yago(args.input_file, relations)
print(f"found {len(facts)} facts for {len(relations)} relations.")
entities = {f[0] for f in facts} | {f[2] for f in facts}
print(f"found {len(entities)} unique entities.")
timestamps = {f[3] for f in facts}
print(f"found {len(timestamps)} unique timestamps.")


os.makedirs(args.output_dir, exist_ok=True)
print(f"writing dataset to {args.output_dir}...", end="")
with open(args.output_dir / "entity2id.json", "w") as f:
    json.dump({entity: i for i, entity in enumerate(entities)}, f)
with open(args.output_dir / "relation2id.json", "w") as f:
    json.dump({rel: i for i, rel in enumerate(relations)}, f)
with open(args.output_dir / "ts2id.json", "w") as f:
    json.dump({ts: i for i, ts in enumerate(timestamps)}, f)
print("done!")

# TODO: split into train.txt/valid.txt/test.txt
# aaaand... we're set?
facts = sorted(facts, key=lambda fact: )
