from typing import Literal, Tuple, List
import argparse, os, json, re
from datetime import date
import pathlib as pl

Fact = Tuple[str, str, str, str]


def string_lstrip(s: str, to_strip: str) -> str:
    try:
        s = s[s.index(to_strip) + len(to_strip) :]
    except ValueError:
        pass
    return s


def clean_prefix(*entities: str) -> List[str]:
    cleaned = []
    for entity in entities:
        entity = string_lstrip(entity, "yago:")
        entity = string_lstrip(entity, "schema:")
        cleaned.append(entity)
    return cleaned


def set_ts(ts: str, val: str, update: Literal["start", "end"]) -> str:
    if ts == "":
        ts = ":"
    if update == "start":
        return val + ":" + ts.split(":")[1]
    else:
        return ts.split(":")[0] + ":" + val


def load_yago(path: pl.Path, relations: set[str], cutoff_year: int) -> set[Fact]:

    facts = {}  # { (subj, rel, obj) => ts }
    print("loading YAGO meta-facts...", end="")
    unparsable_ts_nb = 0
    with open(path / "yago-meta-facts.ntx") as f:
        i = 0

        for line in f:

            if i % 1000000 == 0:
                print(".", end="", flush=True)
            i += 1

            try:
                _, subj, rel, obj, _, metakey, metaval = line.split("\t")
                subj, rel, obj = clean_prefix(subj, rel, obj)
            except ValueError:
                continue

            if not rel in relations:
                continue

            if not (subj, rel, obj) in facts:
                facts[(subj, rel, obj)] = ""

            m = re.match(
                r"\"(-?[0-9]{4}-[0-9]{2}-[0-9]{2})T[0-9:]+Z\"\^\^xsd:dateTime", metaval
            )
            # NOTE: some timestamps are in an incorrect format:
            # - some dates are of the form '_:ID'
            # - two timestamps have a year of 49500
            if m is None:
                unparsable_ts_nb += 1
                continue
            metaval = m.group(1)

            # exclude date that are after a specific year. We
            # try/except since negative dates are not supported by
            # Python's datetime.
            try:
                d = date.fromisoformat(metaval)
                if d.year > cutoff_year:
                    continue
            except ValueError:
                pass

            if metakey == "schema:startDate":
                facts[(subj, rel, obj)] = set_ts(
                    facts[(subj, rel, obj)], metaval, "start"
                )
            elif metakey == "schema:endDate":
                facts[(subj, rel, obj)] = set_ts(
                    facts[(subj, rel, obj)], metaval, "end"
                )

    print("done!")
    print(f"NOTE: there were {unparsable_ts_nb} unparsable timestamps.")

    ts_facts = {key + (ts,) for key, ts in facts.items() if ts != ""}
    print(f"NOTE: dropping {len(facts) - len(ts_facts)} facts without timesamps.")
    return ts_facts


parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", "-i", type=pl.Path)
parser.add_argument("--relations", "-r", default=set(), nargs="*")
parser.add_argument("--output-dir", "-o", type=pl.Path)
parser.add_argument("--cutoff-year", "-c", type=int, default=2024)
args = parser.parse_args()

relations = set(args.relations)
facts = load_yago(args.input_dir, relations, args.cutoff_year)
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


def latest_date(ts: str) -> date:
    """
    .. note::

        Python does not support negative timestamps.  Since
        in our case their relative ordering is non-important (what's
        important is that these timestamps will, in practive, end up
        in the training dataset), we simply treat them as 0001-01-01.

    :param ts: timestamp with a format of either 'START: ':END' or
        'START:END', where START and END are in YYYY-MM-DD format.
    """
    start, end = ts.split(":")
    if end == "":
        if start.startswith("-"):
            return date(1, 1, 1)
        return date.fromisoformat(start)
    if end.startswith("-"):
        return date(1, 1, 1)
    return date.fromisoformat(end)


facts = sorted(facts, key=lambda fact: latest_date(fact[3]))  # type: ignore

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
