from typing import Tuple, List
import argparse, re
import pathlib as pl
import torch
import transformers
from fiction.utils import dump_json, load_facts

# (subj, rel, obj, ts)
Fact = Tuple[str, str, str, str]


def string_lstrip(s: str, to_strip: str) -> str:
    try:
        s = s[s.index(to_strip) + len(to_strip) :]
    except ValueError:
        pass
    return s


def clean_prefix(elt: str) -> str:
    elt = string_lstrip(elt, "yago:")
    elt = string_lstrip(elt, "schema:")
    return elt


def clean_fact_prefix(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_prefix(subj), clean_prefix(rel), clean_prefix(obj), clean_prefix(ts))


def parse_hex_unicode(hex_unicode: str) -> str:
    assert hex_unicode.startswith("u")
    return chr(int(hex_unicode[1:], base=16))


def clean_unicode(elt: str) -> str:
    return re.sub(r"_u[0-9A-E]{4}", lambda m: parse_hex_unicode(m.group()[1:]), elt)


def clean_fact_unicode(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_unicode(subj), rel, clean_unicode(obj), ts)


def clean_underscore(elt: str) -> str:
    elt = re.sub(r"_$", "", elt)
    elt = re.sub(r"_+", " ", elt)
    return elt


def clean_fact_underscore(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_underscore(subj), rel, clean_underscore(obj), ts)


def clean_wiki_id(elt: str) -> str:
    return re.sub(r"Q[0-9]+", "", elt)


def clean_fact_wiki_id(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_wiki_id(subj), rel, clean_wiki_id(obj), ts)


def format_fact(fact: Fact) -> Fact:
    fact = clean_fact_prefix(fact)
    fact = clean_fact_unicode(fact)
    fact = clean_fact_wiki_id(fact)
    fact = clean_fact_underscore(fact)
    return fact


def gen_facts_description(facts: List[Fact], pipeline) -> List[str]:
    """Given list of quadruples FACTS, generate a description using LM.

    :param facts: quadruples for which to generate a description
    :param pipeline: huggingface text-generation pipeline
    """

    prompt = """Given the following event represented as a quadruplet of the form (subject, relation, object, timestamp):
    {}
    Generate a one to three sentences description text for this event, in the style of a newspaper.
    You can add additional details, but the entirety of the information in the given quadruplet must be preserved. 
    Do NOT add any additional information or text: you must only generate the description.
    """

    messages = [
        [
            {
                "role": "system",
                "content": "You are a generation model that is expert at outputting description of events.",
            },
            {"role": "user", "content": prompt.format(format_fact(fact))},
        ]
        for fact in facts
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return [out[0]["generated_text"][-1]["content"] for out in outputs]


def gen_fact_description(fact: Fact, pipeline) -> str:
    """Given the quadruples FACT, generate a description using LM.

    :param fact: quadruple for which to generate a description
    :param pipeline: huggingface text-generation pipeline
    """
    return gen_facts_description([fact], pipeline)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--facts-file",
        type=pl.Path,
        help="file containing facts, one fact per line.",
    )
    parser.add_argument("-o", "--output-file", type=pl.Path, help="output JSON file.")
    parser.add_argument(
        "-l", "--language-model", type=str, default="mistralai/Mistral-7B-v0.1"
    )
    args = parser.parse_args()

    facts = load_facts(args.facts_file, "loading facts")

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.laguage_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    dataset = []
    descs = gen_facts_description(facts, pipeline)
    for fact, desc in zip(facts, descs):
        dataset.append(
            {
                "subject": fact[0],
                "relation": fact[1],
                "object": fact[2],
                "timestamp": fact[3],
                "description": desc,
            }
        )
    dump_json(dataset, args.output_file, "dumping dataset")
