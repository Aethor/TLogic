from typing import Any, List, Optional, Tuple
import json
import pathlib as pl

Fact = Tuple[str, str, str, str]


def dump_json(value: Any, path: pl.Path, progress_msg: Optional[str] = None, **kwargs):
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    with open(path, "w") as f:
        json.dump(value, f, **kwargs)
    if not progress_msg is None:
        print("done!")


def dump_facts(facts: List[Fact], path: pl.Path, progress_msg: Optional[str] = None):
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    with open(path, "w") as f:
        for subj, rel, obj, ts in facts:
            f.write(f"{subj}\t{rel}\t{obj}\t{ts}\n")
    if not progress_msg is None:
        print("done!")


def load_facts(path: pl.Path, progress_msg: Optional[str] = None) -> List[Fact]:
    if not progress_msg is None:
        print(progress_msg + "...", end="")
    facts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            subj, rel, obj, ts = line.split("\t")
            facts.append((subj, rel, obj, ts))
    if not progress_msg is None:
        print("done!")
    return facts
