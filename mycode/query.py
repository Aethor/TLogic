# -*- eval: (code-cells-mode); -*-
# %%
from typing import Literal, Optional, Tuple, List, Dict, TypeVar, Set
from datetime import date, timedelta
import pathlib as pl
import json, random, os, contextlib, sys
import numpy as np
from joblib import Parallel, delayed
from apply import apply_rules
from grapher import Grapher
import rule_application as ra
from score_functions import score_12
from temporal_walk import store_edges

# (subj, rel, obj, ts)
Fact = Tuple[str, str, str, str]

# (subj, rel, ?, ts)
Query = Tuple[str, str, Literal["?"], str]

# [(obj, score), ...]
QueryOutput = List[Tuple[str, float]]


def load_facts(path: pl.Path) -> List[Fact]:
    train_facts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            subj, rel, obj, ts = line.split("\t")
            train_facts.append((subj, rel, obj, ts))
    return train_facts


def load_rules(path: pl.Path, rule_lengths: List[int]) -> Dict[int, dict]:
    with open(path) as f:
        rules_dict = json.load(f)
    rules_dict = {int(k): v for k, v in rules_dict.items()}
    rules_dict = ra.filter_rules(
        rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
    )
    return rules_dict


def make_grapher(
    queries: List[Query],
    train_facts: List[Fact],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    ts2id: Dict[str, int],
) -> Grapher:
    grapher = Grapher.__new__(Grapher)
    grapher.dataset_dir = None
    grapher.entity2id = entity2id
    grapher.relation2id = rel2id.copy()
    counter = len(rel2id)
    for relation in rel2id:
        grapher.relation2id["_" + relation] = counter  # Inverse relation
        counter += 1
    grapher.ts2id = ts2id
    grapher.id2entity = dict([(v, k) for k, v in grapher.entity2id.items()])
    grapher.id2relation = dict([(v, k) for k, v in grapher.relation2id.items()])
    grapher.id2ts = dict([(v, k) for k, v in grapher.ts2id.items()])

    grapher.inv_relation_id = dict()
    num_relations = len(rel2id)
    for i in range(num_relations):
        grapher.inv_relation_id[i] = i + num_relations
    for i in range(num_relations, num_relations * 2):
        grapher.inv_relation_id[i] = i % num_relations

    grapher.train_idx = grapher.add_inverses(grapher.map_to_idx(train_facts))
    # grapher.valid_idx = grapher.add_inverses(grapher.map_to_idx(valid_facts))
    grapher.test_idx = grapher.add_inverses(grapher.map_to_idx(queries))
    grapher.all_idx = np.vstack((grapher.train_idx, grapher.test_idx))
    return grapher


@contextlib.contextmanager
def redirect_stdout_fd(file):
    stdout_fd = sys.stdout.fileno()
    stdout_fd_dup = os.dup(stdout_fd)
    os.dup2(file.fileno(), stdout_fd)
    file.close()
    try:
        yield
    finally:
        os.dup2(stdout_fd_dup, stdout_fd)
        os.close(stdout_fd_dup)


def query(
    queries: List[Query],
    train_facts: List[Fact],
    rules: Dict[int, dict],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
    ts2id: Dict[str, int],
    process_nb: int = 1,
) -> List[QueryOutput]:

    entity2id["?"] = -1

    max_ts_id = max(ts2id.values())
    updated_ts2id = ts2id.copy()
    for i, ts in enumerate(set(q[3] for q in queries)):
        updated_ts2id[ts] = max_ts_id + i

    grapher = make_grapher(queries, train_facts, entity2id, rel2id, updated_ts2id)

    id2entity = {v: k for k, v in entity2id.items()}
    with open(os.devnull, "w") as devnull:
        with redirect_stdout_fd(devnull):
            if process_nb == 1:
                scores, _ = apply_rules(
                    grapher.test_idx,
                    rules,
                    grapher,
                    store_edges(grapher.train_idx),
                    score_12,
                    20,  # top_k
                    0,
                    len(grapher.test_idx),
                    [[0.1, 0.5]],  # args for score_12
                    0,  # window
                )
                answers = [[] for _ in range(len(queries))]
                for answer_i, scores in scores[0].items():
                    try:
                        answers[answer_i] = [
                            (id2entity[k], v) for k, v in scores.items()
                        ]
                    except IndexError:
                        continue
                return answers
            else:
                queries_nb = len(grapher.test_idx) // process_nb
                poutput = Parallel(n_jobs=process_nb)(
                    delayed(apply_rules)(
                        grapher.test_idx,
                        rules,
                        grapher,
                        store_edges(grapher.train_idx),
                        score_12,
                        20,  # top_k
                        i,
                        queries_nb,
                        [[0.1, 0.5]],  # args for score_12
                        0,  # window
                    )
                    for i in range(process_nb)
                )

        answers = [[] for _ in range(len(queries))]
        for i in range(process_nb):
            for answer_i, scores in poutput[i][0][0].items():
                try:
                    answers[answer_i] = [(id2entity[k], v) for k, v in scores.items()]
                except IndexError:
                    continue

        return answers


# For each entity, we sample at most n relations for which we make
# request to TLogic to predict their object. Since we want to keep our
# database consistent, when generating a quadruplet, we add it to the
# database, for future generation, forcing us to generate quadruplet
# sequentially. We generate a random date during 2026.

# We remark that, in most cases, start/end pairs are exclusive (only a
# work/spouse/team at the same time)
# 1. if an entity has a startX but no related endX, we see if we can
# generate endX
# 2. if an entity does not have a an active startX, we generate startX
T = TypeVar("T")


def maybe_max(lst: List[T], **kwargs) -> Optional[T]:
    if len(lst) == 0:
        return None
    return max(lst, **kwargs)


def rel_is_active(rel: str, entity_facts: List[Fact]) -> bool:
    latest_start = maybe_max([f[3] for f in entity_facts if f[1] == rel])
    if latest_start is None:
        return False
    endRel = "end" + rel[5:]
    latest_end = maybe_max([f[3] for f in entity_facts if f[1] == endRel])
    if latest_end is None:
        return False
    return date.fromisoformat(latest_start) > date.fromisoformat(latest_end)


def sample_new_fact(
    entity: str,
    entity_facts: List[Fact],
    ts: str,
    train_facts: List[Fact],
    entity2id,
    rel2id,
    ts2id,
) -> Optional[Fact]:
    future_rels_candidate = []
    for rel in rel2id:
        if rel.startswith("start"):
            if not rel_is_active(rel, entity_facts):
                future_rels_candidate.append(rel)
        elif rel.startswith("end"):
            if rel_is_active(rel, entity_facts):
                future_rels_candidate.append(rel)
        else:
            future_rels_candidate.append(rel)
    answers = query(
        [(entity, rel, "?", ts) for rel in future_rels_candidate],
        train_facts,
        rules,
        entity2id,
        rel2id,
        ts2id,
    )
    if all(len(ans) == 0 for ans in answers):
        return None
    candidates = [c for c in answers if len(c) > 0]
    candidate_i = random.randint(0, len(candidates) - 1)
    obj = candidates[candidate_i][0][0]
    rel = future_rels_candidate[candidate_i]
    return (entity, rel, obj, ts)


# %% Load everything from disk
rules = load_rules(
    pl.Path("../output/yago4.5/060525112719_r[1,2,3]_n100_exp_s12_rules.json"),
    [1, 2, 3],
)
train_facts = (
    load_facts(pl.Path("../data/yago4.5/train.txt"))
    + load_facts(pl.Path("../data/yago4.5/valid.txt"))
    + load_facts(pl.Path("../data/yago4.5/test.txt"))
)
with open("../data/yago4.5/entity2id.json") as f:
    entity2id = json.load(f)
with open("../data/yago4.5/relation2id.json") as f:
    rel2id = json.load(f)
with open("../data/yago4.5/ts2id.json") as f:
    ts2id = json.load(f)


# %% Generate new triplets
subj_entities = list(set([f[0] for f in train_facts]))
new_facts = []
n = 1000  # TODO:
d = date(2026, 1, 1)
while d.year < 2027:
    ts = d.strftime("%Y-%m-%d")
    new_fact = None
    tries = 0
    print(f"generating a fact for {ts}...", end="")
    while new_fact is None or tries >= 1000:
        entity = random.choice(subj_entities)
        entity_facts = [f for f in train_facts if f[0] == entity]
        new_fact = sample_new_fact(
            entity, entity_facts, ts, train_facts, entity2id, rel2id, ts2id
        )
        tries += 1
    if new_fact is None:
        print(f"it felt impossible to generate a new fact for {ts}, I give up.")
    else:
        train_facts.append(new_fact)
        ts2id[ts] = max(ts2id.values()) + 1
        print(new_fact)
    d = d + timedelta(days=1)

# %% Database schema experiments
from copy import deepcopy
from yagottl.ttl import Graph

schema = Graph()
schema.loadTurtleFile("../yago4.5-mini/yago-schema.ttl", "loading YAGO schema")

taxonomy = Graph()
taxonomy.loadTurtleFile("../yago4.5-mini/yago-taxonomy.ttl", "loading YAGO taxonomy")

facts = Graph()
facts.loadTurtleFile("../yago4.5-mini/yago-facts.ttl", "loading cold facts")


# %%
# let's suppose we get a fact (subj, rel, obj, ts)
# how do we go about checking that it is valid?
#
# 1. the REL should be valid for SUBJ (This can actually be checked
#    before the query)
# 2. the OBJ should be valid for REL.
#
# to check (say for 1.):
#
# 1. get the rdf:types of SUBJ
# 2. also get all supertypes
# 3. check all of their sh:property. In particular, a sh:property
#    should have sh:path with REL
def allowed_props(subj: str, facts: Graph, schema: Graph, taxonomy: Graph) -> Set[str]:
    allowed_properties = set()
    types = deepcopy(facts.index[subj]["rdf:type"])
    while len(types) > 0:
        typ = types.pop()
        # add all supertype to the search
        types |= taxonomy.index.get(typ, {}).get("rdfs:subClassOf", set())
        try:
            type_attrs = schema.index[typ]
        except KeyError:
            continue
        for prop in type_attrs.get("sh:property", set()):
            for prop_path in schema.index[prop]["sh:path"]:
                allowed_properties.add(prop_path)
    return allowed_properties


# let's work with the following example:
fact = ("yago:George_Washington", "schema:award", "yago:Juilliard_School", "2026-12-31")
allowed_props(fact[0], facts, schema, taxonomy)
allowed_props("yago:Oslo", facts, schema, taxonomy)
