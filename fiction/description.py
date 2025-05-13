from typing import Tuple, List
import argparse, re
import pathlib as pl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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


def gen_facts_description(facts: List[Fact], lm, tokenizer) -> List[str]:
    """Given list of quadruples FACTS, generate a description using LM.

    :param facts: quadruples for which to generate a description
    :param lm: language model to use for generation with :func:`generate`
    :param tokenizer: huggingface tokenizer for lm
    """
    prompt = """Given an event represented as a quadruple (subject, relation, object, timestamp), generate a description text for this event of around 4 to 5 sentences. You can add additional invented details, but the information in the given quadruplet must be preserved. Only answer with the description, do not add any other output. Here is an example of this task:

Input:
('Linus Torvalds', 'startWorksFor', 'Microsoft', '2026-01-01')

Output:
A truly significant and unexpected development is poised to reshape the tech landscape at the dawn of the new year. On January 1, 2026, the iconic creator of the Linux kernel, Linus Torvalds, is set to officially join Microsoft. This surprising move sees the figurehead of open-source collaboration bringing his unparalleled expertise to the Redmond giant, a company that has increasingly embraced Linux and open source in recent years. While his exact role remains a subject of much speculation, his presence at Microsoft is anticipated to greatly influence their cloud strategies and further solidify their commitment to the open-source ecosystem, marking a historic moment in the industry.

Input:
{}

Output:
    """
    prompts = [prompt.format(format_fact(fact)) for fact in facts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer(prompts, return_tensors="pt").to(device)
    ids = lm.generate(**model_inputs)
    decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
    # strip input
    decoded = [
        d[model_inputs["input_ids"][i].shape[1] :] for i, d in enumerate(decoded)
    ]
    return decoded


def gen_fact_description(fact: Fact, lm, tokenizer) -> str:
    """Given the quadruples FACT, generate a description using LM.

    :param fact: quadruple for which to generate a description
    :param lm: language model to use for generation with :func:`generate`
    :param tokenizer: huggingface tokenizer for lm
    """
    return gen_facts_description([fact], lm, tokenizer)[0]


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

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.language_model,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, padding_side="left")

    dataset = []
    for fact in facts:
        print(fact)
        desc = gen_fact_description(fact, lm, tokenizer)
        print(desc)
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
