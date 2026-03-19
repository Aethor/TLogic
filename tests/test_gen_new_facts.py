import re
from hypothesis import given, strategies as st
from fiction.gen_new_facts import Fact, relation_is_coherent


@st.composite
def st_timestamp(draw):
    return draw(st.dates()).isoformat()


st_ascii = st.text(st.characters(codec="ascii"))
st_subj = st_ascii
st_obj = st_ascii
st_rel = st.from_regex(r"(schema|yago):(start|end)[A-Z].*", fullmatch=True)
st_start_rel = st.from_regex(r"(schema|yago):start[A-Z].*", fullmatch=True)
st_end_rel = st.from_regex(r"(schema|yago):end[A-Z].*", fullmatch=True)


def to_start_rel(end_rel: str) -> str:
    return re.sub(r"([^:]+):(end)(.+)", r"\1:start\3", end_rel)


def to_end_rel(start_rel: str) -> str:
    return re.sub(r"([^:]+):(start)(.+)", r"\1:end\3", start_rel)


@st.composite
def st_fact(draw, subj=st_subj, rel=st_rel, obj=st_obj, timestamp=st_timestamp()):
    return (draw(subj), draw(rel), draw(obj), draw(timestamp))


@given(st_fact())
def test_relation_is_not_coherent_with_itself(fact: Fact):
    _, rel, obj, _ = fact
    assert not relation_is_coherent(rel, obj, [fact])


@given(st_start_rel, st_obj)
def test_no_relation_is_coherent(start_rel: str, obj: str):
    assert relation_is_coherent(start_rel, obj, [])


@given(st_start_rel, st_obj, st.lists(st_fact(subj=st.just("A")), min_size=1))
def test_open_relation_is_coherent(start_rel: str, obj: str, facts: list[Fact]):
    last_ts = max(ts for _, _, _, ts in facts)
    assert relation_is_coherent(
        to_end_rel(start_rel), obj, facts + [("A", start_rel, obj, last_ts)]
    )
