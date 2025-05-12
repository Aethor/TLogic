from typing import Tuple, List
import argparse
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


def clean_fact(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_prefix(subj), clean_prefix(rel), clean_prefix(obj), clean_prefix(ts))


def gen_facts_description(facts: List[Fact], lm, tokenizer) -> str:
    """Given list of quadruples FACTS, generate a description using LM.

    :param facts: quadruples for which to generate a description
    :param lm: language model to use for generation with :func:`generate`
    :param tokenizer: huggingface tokenizer for lm
    """
    prompt = """Given the following event represented as a quadruple of the form (subject, relation, object, timestamp):
{}
Generate a description text for this event of around 4 to 5 sentences. You can add additional invented details, but the information in the given quadruplet must be preserved. Only answer with the description, do not add any other output."""
    prompts = [prompt.format(clean_fact(fact)) for fact in facts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer(prompts, return_tensors="pt").to(device)
    ids = lm.generate(**model_inputs)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


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
