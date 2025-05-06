from typing import Literal, Tuple, List
import argparse, os, json, re
from datetime import date
from collections import Counter
import pathlib as pl

Fact = Tuple[str, str, str, str]


class Date:
    """
    .. note::

        Python datetime.date does not support negative timestamps.  Since
        in our case ordering is important, we create his dummy class for sorting
    """

    @staticmethod
    def parse_yyyymmdd(ts: str) -> Tuple[int, int, int]:
        is_negative = False
        if ts.startswith("-"):
            ts = ts[1:]
            is_negative = True
        year, month, day = map(int, ts.split("-"))
        if is_negative:
            year = -year
        return (year, month, day)

    def __init__(self, ts: str):
        """
        :param ts: timestamp with a format of either 'DATE', 'START:',
            ':END' or 'START:END', where START and END are in
            YYYY-MM-DD format.
        """
        self.ts = ts

        if not ":" in ts:
            self.year, self.month, self.day = Date.parse_yyyymmdd(ts)
            self.sort_year, self.sort_month, self.sort_day = (
                self.year,
                self.month,
                self.day,
            )
        else:
            start, end = ts.split(":")
            if not start == "":
                self.start_year, self.start_month, self.start_day = Date.parse_yyyymmdd(
                    start
                )
                self.sort_year, self.sort_month, self.sort_day = (
                    self.start_year,
                    self.start_month,
                    self.start_day,
                )
            if not end == "":
                self.end_year, self.end_month, self.end_day = Date.parse_yyyymmdd(end)
                self.sort_year, self.sort_month, self.sort_day = (
                    self.end_year,
                    self.end_month,
                    self.end_day,
                )

    def __lt__(self, other) -> bool:
        if self.sort_year != other.sort_year:
            return self.sort_year < other.sort_year
        if self.sort_month != other.sort_month:
            return self.sort_month < other.sort_month
        if self.sort_day != other.sort_day:
            return self.sort_day < other.sort_day
        return False


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


def load_yago(
    path: pl.Path, relations: set[str], min_year: int, max_year: int
) -> set[Fact]:

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

            d = Date(metaval)
            if d.year > max_year or d.year < min_year:
                continue

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


def linearize_facts(facts: List[Fact]) -> List[Fact]:

    linearized_facts = []

    print("linearizing facts...", end="")
    for subj, rel, obj, ts in facts:
        if not ":" in ts:
            linearized_facts.append((subj, rel, obj, ts))
        elif ts.endswith(":"):
            linearized_facts.append((subj, f"start{rel.capitalize()}", obj, ts[:-1]))
        elif ts.startswith(":"):
            linearized_facts.append((subj, f"end{rel.capitalize()}", obj, ts[1:]))
        else:
            start, end = ts.split(":")
            if start == end:
                linearized_facts.append((subj, rel, obj, start))
            else:
                linearized_facts.append((subj, f"start{rel.capitalize()}", obj, start))
                linearized_facts.append((subj, f"end{rel.capitalize()}", obj, end))
    print("done!")

    assert len(linearized_facts) >= len(facts)
    return linearized_facts


def sparsity_filter(facts: List[Fact], threshold: int, depth: int = 0) -> List[Fact]:

    print(f"reducing sparsity (round {depth+1})...", end="")
    counter = Counter()
    for subj, _, obj, _ in facts:
        counter[subj] += 1
        counter[obj] += 1

    filtered_facts = []
    for fact in facts:
        subj, _, obj, _ = fact
        if counter[subj] <= threshold or counter[obj] <= threshold:
            continue
        filtered_facts.append(fact)
    print(f"done! (removed {len(facts) - len(filtered_facts)} entities)")

    # if we removed facts, other entities might have a degree of 1:
    # call recursively to fix these
    if len(filtered_facts) < len(facts):
        filtered_facts = sparsity_filter(filtered_facts, threshold, depth + 1)

    assert len(filtered_facts) <= len(facts)
    return filtered_facts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=pl.Path)
    parser.add_argument("--relations", "-r", default=set(), nargs="*")
    parser.add_argument("--output-dir", "-o", type=pl.Path)
    parser.add_argument("--min-year", "-miny", type=int, default=2024)
    parser.add_argument("--max-year", "-maxy", type=int, default=1925)
    parser.add_argument("--sparsity-filter-threshold", "-s", type=int, default=3)
    parser.add_argument("--linearize", "-l", action="store_true")
    args = parser.parse_args()

    relations = set(args.relations)
    facts = list(load_yago(args.input_dir, relations, args.min_year, args.max_year))
    if args.linearize:
        facts = linearize_facts(facts)
        relations = set(f[1] for f in facts)
    facts = sparsity_filter(facts, args.sparsity_filter_threshold)
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
        json.dump({ts: i for i, ts in enumerate(sorted(timestamps, key=Date))}, f)
    print("done!")

    facts = sorted(facts, key=lambda fact: Date(fact[3]))  # type: ignore

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
