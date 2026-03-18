from __future__ import annotations
from typing import Optional, Protocol, Callable
import argparse, re, random, json, subprocess
import pathlib as pl
from datetime import datetime, timedelta
from dataclasses import dataclass
import torch
import requests
from transformers import pipeline  # type: ignore
from transformers.pipelines.base import Pipeline
from tqdm import tqdm
import numpy as np
from more_itertools import flatten
from sklearn.cluster import AgglomerativeClustering
from openai import OpenAI
from fiction.yagottl.TurtleUtils import YagoDBInfo
from fiction.yagottl.schema import facts_dist
from fiction.utils import dump_json, load_facts
from fiction.yago_rel_desc import YAGO_REL_DESC

# (subj, rel, obj, ts)
Fact = tuple[str, str, str, str]


@dataclass
class GCloudConfig:
    project: str
    location: str
    api_endpoint: str

    @staticmethod
    def from_json(json_str: str) -> GCloudConfig:
        return GCloudConfig(**json.loads(json_str))


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
    assert hex_unicode.lower().startswith("u")
    return chr(int(hex_unicode[1:], base=16))


def clean_unicode(elt: str) -> str:
    return re.sub(r"_[uU][0-9A-E]{4}", lambda m: parse_hex_unicode(m.group()[1:]), elt)


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


def clean_generic_instance(elt: str) -> str:
    return re.sub(r" ?generic instance", "", elt, flags=re.IGNORECASE)


def clean_fact_generic_instance(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    return (clean_generic_instance(subj), rel, clean_generic_instance(obj), ts)


def format_fact(fact: Fact) -> Fact:
    fact = clean_fact_prefix(fact)
    fact = clean_fact_unicode(fact)
    fact = clean_fact_wiki_id(fact)
    fact = clean_fact_underscore(fact)
    fact = clean_fact_generic_instance(fact)
    return fact


def group_related_facts(
    facts: list[Fact],
    min_size: int,
    max_size: int,
    db_info: YagoDBInfo,
    alpha: float = 0.9,
    k: float = 0.03,
) -> list[list[Fact]]:
    """Group related facts, returning a list of groups of such facts"""
    dists = np.zeros((len(facts), len(facts)))
    for i in tqdm(range(len(facts)), desc="dist"):
        for j in range(i):
            dist = facts_dist(facts[i], facts[j], alpha, k, db_info)
            dists[i][j] = dist
            dists[j][i] = dist

    clustering = AgglomerativeClustering(
        metric="precomputed", linkage="average", distance_threshold=0.5, n_clusters=None
    ).fit(dists)
    clusters_nb = len(set(clustering.labels_))
    clusters = [[[]] for _ in range(clusters_nb)]
    for fact, label in zip(facts, clustering.labels_):
        if len(clusters[label][-1]) < max_size:
            clusters[label][-1].append(fact)
        else:
            # max size of this cluster has been reached: create a new
            # one
            clusters[label].append([])
    # flatten nested clusters, filter for min_size
    return [c for c in flatten(clusters) if len(c) >= min_size]


def make_get_styled_multifact_prompt(
    style: str, additional_instructions: Optional[str] = None
) -> Callable[[list[Fact]], str]:
    def get_multifact_prompt(fact_group: list[Fact]) -> str:
        formatted_facts = [format_fact(fact) for fact in fact_group]
        formatted_facts = [randomize_fact_ts_style(fact) for fact in formatted_facts]
        formatted_facts = "\n".join(str(fact) for fact in formatted_facts)

        relations = {rel for _, rel, _, _ in formatted_facts}
        relations = "\n".join(f"{rel}: {YAGO_REL_DESC.get(rel)}" for rel in relations)

        prompt = f"""Given the following events represented as quadruplets of the form (subject, relation, object, timestamp):
        {formatted_facts}
        and the following definitions for the relations:
        {relations}
        Generate a short paragraph describing these events, in the style of {style}. The entirety of the information in the given quadruplets must be preserved. Do NOT add any additional information or text: you must only generate the description.
        """
        if not additional_instructions is None:
            prompt += additional_instructions

        if random.random() < 0.25:
            dates = sorted(
                [datetime.strptime(ts, "%Y-%m-%d") for _, _, _, ts in fact_group]
            )
            min_date = dates[0] - timedelta(days=random.randint(0, 7))
            max_date = dates[0] + timedelta(days=random.randint(0, 7))
            delta = max_date - min_date
            current_date = min_date + timedelta(random.randint(0, delta.days))
            current_date = randomize_ts_style(current_date)
            prompt += f" The current date is {current_date}. In addition to the date of the event, indicate the current date at the top of your text while respecting the style of the document."

        return prompt

    return get_multifact_prompt


get_wiki_multifact_prompt = make_get_styled_multifact_prompt("a wikipedia article")
get_twitter_multifact_prompt = make_get_styled_multifact_prompt(
    "a tweet",
    "You can add hashtags, but do no explicitly add the relation as a hashtag.",
)
get_news_multifact_prompt = make_get_styled_multifact_prompt("a news article")
get_blog_multifact_prompt = make_get_styled_multifact_prompt("a blog post")


def get_multifact_prompt(fact_group: list[Fact]) -> str:
    get_multifact_prompt_fn = random.choice(
        [
            get_news_multifact_prompt,
            get_wiki_multifact_prompt,
            get_twitter_multifact_prompt,
            get_blog_multifact_prompt,
        ]
    )
    return get_multifact_prompt_fn(fact_group)


def ts_day_ordinal(day: int) -> str:
    ord_suffix = (
        "th" if 10 <= day <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return str(day) + ord_suffix


def randomize_ts_style(d: datetime) -> str:
    day_style = random.choice(["%B ~d, %Y", "%Y-%m-%d"])
    weekday_style = random.choice(["%A, ", "%a, ", ""])
    style = weekday_style + day_style
    return d.strftime(style).replace("~d", ts_day_ordinal(d.day))


def randomize_fact_ts_style(fact: Fact) -> Fact:
    subj, rel, obj, ts = fact
    d = datetime.strptime(ts, "%Y-%m-%d")
    new_ts = randomize_ts_style(d)
    return (subj, rel, obj, new_ts)


def make_get_styled_fact_prompt(
    style: str, additional_instructions: Optional[str] = None
) -> Callable[[Fact], str]:
    def get_styled_fact_prompt(fact: Fact) -> str:
        formatted_fact = format_fact(fact)
        formatted_fact = randomize_fact_ts_style(formatted_fact)

        formatted_relation = formatted_fact[1]

        prompt = f"""Given the following event represented as a quadruplet of the form (subject, relation, object, timestamp):
        {formatted_fact},
        The following definition for the {formatted_relation} relation:
        {YAGO_REL_DESC.get(formatted_relation)},
        Generate a one to three sentences description text for this event, in the style of {style}. The entirety of the information in the given quadruplet must be preserved. You can add additional information, but do NOT add any additional temporal event.
        """
        if not additional_instructions is None:
            prompt += additional_instructions

        if random.random() < 0.25:
            d = datetime.strptime(fact[3], "%Y-%m-%d")
            current_date = d + timedelta(days=random.randint(-7, 7))
            current_date = randomize_ts_style(current_date)
            prompt += f" The current date is {current_date}. In addition to the date of the event, indicate the current date at the top of your text while respecting the style of the document."

        return prompt

    return get_styled_fact_prompt


get_news_fact_prompt = make_get_styled_fact_prompt("a news article")
get_wiki_fact_prompt = make_get_styled_fact_prompt("a wikipedia article")
get_twitter_fact_prompt = make_get_styled_fact_prompt(
    "a tweet",
    "You can add hashtags, but do no explicitly add the relation as a hashtag.",
)
get_blog_fact_prompt = make_get_styled_fact_prompt("a blog post")


def get_fact_prompt(fact: Fact) -> str:
    get_fact_prompt_fn = random.choice(
        [
            get_news_fact_prompt,
            get_wiki_fact_prompt,
            get_twitter_fact_prompt,
            get_blog_fact_prompt,
        ]
    )
    return get_fact_prompt_fn(fact)


class DescriptionGenerator(Protocol):
    def gen_facts_description(self, facts: list[Fact]) -> list[Optional[str]]: ...

    def gen_fact_description(self, fact: Fact) -> Optional[str]:
        return self.gen_facts_description([fact])[0]

    def gen_multifacts_description(
        self, fact_groups: list[list[Fact]]
    ) -> list[Optional[str]]: ...

    def gen_multifact_description(self, fact_group: list[Fact]) -> Optional[str]:
        return self.gen_multifacts_description([fact_group])[0]


class HuggingfaceDescriptionGenerator(DescriptionGenerator):
    def __init__(self, huggingface_id: str, batch_size: int = 8):
        self.pipe = pipeline(
            "text-generation",
            model=huggingface_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        assert not self.pipe.tokenizer is None
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.pipe.tokenizer.padding_side = "left"
        self.batch_size = batch_size

    def gen_facts_description(self, facts: list[Fact]) -> list[Optional[str]]:
        """Given list of quadruples FACTS, generate a description using
        PIPE.

        :param facts: quadruples for which to generate a description
        :param pipe: huggingface text-generation pipeline
        """
        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a generation model that is expert at outputting description of events.",
                },
                {"role": "user", "content": get_fact_prompt(fact)},
            ]
            for fact in facts
        ]

        descriptions = []
        for batch_start in tqdm(range(0, len(messages), self.batch_size)):
            batch_end = batch_start + self.batch_size
            batch_messages = messages[batch_start:batch_end]
            batch_descriptions = ["" for _ in batch_messages]
            batch_years = [
                str(datetime.strptime(ts, "%Y-%m-%d").year)
                for _, _, _, ts in facts[batch_start:batch_end]
            ]
            has_years = False
            while not has_years:
                # only perform generation for description that don't have
                # fact year yet
                batch_indices = [
                    i
                    for i in range(len(batch_messages))
                    if not batch_years[i] in batch_descriptions[i]
                ]
                outputs = self.pipe(
                    [batch_messages[i] for i in batch_indices],
                    max_new_tokens=256,
                    pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
                    batch_size=len(batch_indices),
                )
                for i, output in enumerate(outputs):  # type: ignore
                    desc = output[0]["generated_text"][-1]["content"]  # type: ignore
                    batch_descriptions[batch_indices[i]] = desc  # type: ignore
                has_years = all(
                    year in desc for year, desc in zip(batch_years, batch_descriptions)
                )
            descriptions += batch_descriptions

        return descriptions

    def gen_multifacts_description(
        self, fact_groups: list[list[Fact]]
    ) -> list[Optional[str]]:
        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a generation model that is expert at outputting description of events.",
                },
                {
                    "role": "user",
                    "content": get_multifact_prompt(fact_group),
                },
            ]
            for fact_group in fact_groups
        ]

        descriptions = []
        for batch_start in tqdm(range(0, len(messages), self.batch_size)):
            batch_end = batch_start + self.batch_size
            batch_messages = messages[batch_start:batch_end]
            batch_descriptions = ["" for _ in batch_messages]
            for facts in fact_groups[batch_start:batch_end]:
                assert (
                    len({datetime.strptime(fact[3], "%Y-%m-%d").year for fact in facts})
                    == 1
                )
            batch_years = [
                str(datetime.strptime(facts[0][3], "%Y-%m-%d").year)
                for facts in fact_groups[batch_start:batch_end]
            ]
            has_years = False
            while not has_years:
                # only perform generation for description that don't have
                # fact year yet
                batch_indices = [
                    i
                    for i in range(len(batch_messages))
                    if not batch_years[i] in batch_descriptions[i]
                ]
                outputs = self.pipe(
                    [batch_messages[i] for i in batch_indices],
                    max_new_tokens=256,
                    pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
                    batch_size=len(batch_indices),
                )
                for i, output in enumerate(outputs):  # type: ignore
                    desc = output[0]["generated_text"][-1]["content"]  # type: ignore
                    batch_descriptions[batch_indices[i]] = desc  # type: ignore
                has_years = all(
                    year in desc for year, desc in zip(batch_years, batch_descriptions)
                )
            descriptions += batch_descriptions

        return descriptions


class OpenAIAPIDescriptionGenerator(DescriptionGenerator):
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def gen_facts_description(self, facts: list[Fact]) -> list[Optional[str]]:
        descriptions = []
        usage_stats = []

        for fact in tqdm(facts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": get_fact_prompt(fact)}],
                    timeout=30,
                )
            except Exception as e:
                tqdm.write(
                    f"warning: could not generate a description for {fact}. (reason: {e})"
                )
                descriptions.append(None)
                continue

            desc = response.choices[0].message.content
            descriptions.append(desc)
            if response.usage:
                usage_stats.append(
                    {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                    }
                )

        if usage_stats:
            print("usage summary:")
            print(
                "completions_tokens: {}".format(
                    sum(s["completion_tokens"] for s in usage_stats)
                )
            )
            print(
                "prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats))
            )

        return descriptions

    def gen_multifacts_description(
        self, fact_groups: list[list[Fact]]
    ) -> list[Optional[str]]:
        descriptions = []
        usage_stats = []

        for fact_group in tqdm(fact_groups):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": get_multifact_prompt(fact_group)}
                    ],
                    timeout=30,
                )
            except Exception as e:
                tqdm.write(
                    f"warning: could not generate a description for {fact_group}. (reason: {e})"
                )
                descriptions.append(None)
                continue

            desc = response.choices[0].message.content
            descriptions.append(desc)
            if response.usage:
                usage_stats.append(
                    {
                        "completion_tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                    }
                )

        if usage_stats:
            print("usage summary:")
            print(
                "completions_tokens: {}".format(
                    sum(s["completion_tokens"] for s in usage_stats)
                )
            )
            print(
                "prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats))
            )

        return descriptions


class VertexAIDescriptionGenerator(DescriptionGenerator):
    def __init__(self, config: GCloudConfig, model_id: str):
        self.config = config
        self.model_id = model_id

    def gen_facts_description(self, facts: list[Fact]) -> list[Optional[str]]:
        url = f"https://{self.config.api_endpoint}/v1/projects/{self.config.project}/locations/{self.config.location}/endpoints/openapi/chat/completions"

        descriptions = []
        usage_stats = []

        for fact in tqdm(facts):
            access_token = (
                subprocess.check_output(["gcloud", "auth", "print-access-token"])
                .decode("utf8")
                .strip("\n")
            )
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model_id,
                "stream": False,
                "messages": [{"role": "user", "content": get_fact_prompt(fact)}],
            }

            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
            except Exception as e:
                tqdm.write(
                    f"warning: could not generate a description for {fact}. (reason: {e})"
                )
                descriptions.append(None)
                continue
            if response.status_code != 200:
                tqdm.write(
                    f"warning: could not generate a description for {fact}. (reason: {response.status_code} {response.json()})"
                )
                descriptions.append(None)
                continue
            response_json = response.json()
            desc = response_json["choices"][0]["message"]["content"]
            descriptions.append(desc)
            usage_stats.append(response_json["usage"])

        print("usage summary:")
        print(
            "completions_tokens: {}".format(
                sum(s["completion_tokens"] for s in usage_stats)
            )
        )
        print("prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats)))

        return descriptions

    def gen_multifacts_description(
        self, fact_groups: list[list[Fact]]
    ) -> list[Optional[str]]:
        url = f"https://{self.config.api_endpoint}/v1/projects/{self.config.project}/locations/{self.config.location}/endpoints/openapi/chat/completions"

        descriptions = []
        usage_stats = []

        for fact_group in tqdm(fact_groups):
            access_token = (
                subprocess.check_output(["gcloud", "auth", "print-access-token"])
                .decode("utf8")
                .strip("\n")
            )
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            data = {
                "model": self.model_id,
                "stream": False,
                "messages": [
                    {"role": "user", "content": get_multifact_prompt(fact_group)}
                ],
            }

            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
            except Exception as e:
                tqdm.write(
                    f"warning: could not generate a description for {fact_group}. (reason: {e})"
                )
                descriptions.append(None)
                continue
            if response.status_code != 200:
                tqdm.write(
                    f"warning: could not generate a description for {fact_group}. (reason: {response.status_code} {response.json()})"
                )
                descriptions.append(None)
                continue
            response_json = response.json()
            desc = response_json["choices"][0]["message"]["content"]
            descriptions.append(desc)
            usage_stats.append(response_json["usage"])

        print("usage summary:")
        print(
            "completions_tokens: {}".format(
                sum(s["completion_tokens"] for s in usage_stats)
            )
        )
        print("prompt_tokens: {}".format(sum(s["prompt_tokens"] for s in usage_stats)))

        return descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="If a --multi-* argument is specified, all --multi-* arguments must be specified."
    )
    parser.add_argument(
        "-f",
        "--facts-file",
        type=pl.Path,
        help="file containing facts, one fact per line.",
    )
    parser.add_argument(
        "-mn",
        "--multi-min-size",
        type=int,
        default=None,
        help="Min size for multi-facts generation.",
    )
    parser.add_argument(
        "-mx",
        "--multi-max-size",
        type=int,
        default=None,
        help="Max size for multi-facts generation.",
    )
    parser.add_argument(
        "-my",
        "--multi-yago-dir",
        type=pl.Path,
        default=None,
        help="Yago directory for multi-facts generation.",
    )
    parser.add_argument(
        "-ma",
        "--multi-alpha",
        type=float,
        default=None,
        help="alpha in fact similarity computation for multi-facts generation.",
    )
    parser.add_argument(
        "-mk",
        "--multi-k",
        type=float,
        default=None,
        help="k for similarity computation in multi-facts generation.",
    )
    parser.add_argument("-o", "--output-file", type=pl.Path, help="output JSON file.")
    parser.add_argument(
        "-l",
        "--language-model",
        type=str,
        default="hf:meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace ID of the language model used to generate descriptions, prefixed by 'hf:' (example: 'hf:meta-llama/Meta-Llama-3.1-8B-Instruct'). Alternatively, the ID of a Google Vertex AI model, prefixed by 'vertexai:' (example: 'vertexai:meta/llama-3.3-70b-instruct-maas'). In that case, you must also set --gcloud-config. Alternatively, use the OpenAI API with prefix 'openai:' (example: 'openai:gpt-4'). In that case, you must also set --openai-api-key and optionally --openai-base-url for services like OpenRouter.",
    )
    parser.add_argument(
        "-g",
        "--gcloud-config",
        type=str,
        default="{}",
        help='google cloud config, as a json dictionary. The following keys must be present: "project", "location", "api_endpoint". Example: {"project": "your_project_id", "location": "us-central1", "api_endpoint": "us-central1-aiplatform.googleapis.com"}. Note that the access token will be dynamically obtained with $(gcloud auth print-access-token), so make sure you configured your gcloud CLI accordingly.',
    )
    parser.add_argument(
        "-k",
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key. Required when using 'openai:' prefix in --language-model. Can be used with OpenRouter or other OpenAI-compatible services.",
    )
    parser.add_argument(
        "-b",
        "--openai-base-url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible API. Optional. Use this for services like OpenRouter (e.g., 'https://openrouter.ai/api/v1').",
    )
    args = parser.parse_args()

    facts = load_facts(args.facts_file, "loading facts")

    lm_provider, lm = args.language_model.split(":")

    dataset = []
    if args.multi_min_size:  # all --multi arguments should be specified
        db_info = YagoDBInfo.from_yago_dir(args.multi_yago_dir)
        fact_groups = group_related_facts(
            facts,
            args.multi_min_size,
            args.multi_max_size,
            db_info,
            alpha=args.multi_alpha,
            k=args.multi_k,
        )
        if lm_provider == "hf":
            description_generator = HuggingfaceDescriptionGenerator(lm)
        elif lm_provider == "vertexai":
            gconfig = GCloudConfig.from_json(args.gcloud_config)
            description_generator = VertexAIDescriptionGenerator(gconfig, lm)
        elif lm_provider == "openai":
            description_generator = OpenAIAPIDescriptionGenerator(
                lm, args.openai_api_key, args.openai_base_url
            )
        else:
            raise ValueError(f"Unknown LLM provider: {lm_provider}.")
        descs = description_generator.gen_multifacts_description(fact_groups)
        for fact_group, desc in zip(fact_groups, descs):
            if desc is None:
                desc = ["None", "None", "None", "None"]
            dataset.append(
                {
                    "facts": [
                        {
                            "subject": fact[0],
                            "relation": fact[1],
                            "object": fact[2],
                            "timestamp": fact[3],
                        }
                        for fact in fact_group
                    ],
                    "description": desc,
                }
            )
    else:
        if lm_provider == "hf":
            description_generator = HuggingfaceDescriptionGenerator(lm)
        elif lm_provider == "vertexai":
            gconfig = GCloudConfig.from_json(args.gcloud_config)
            description_generator = VertexAIDescriptionGenerator(gconfig, lm)
        elif lm_provider == "openai":
            description_generator = OpenAIAPIDescriptionGenerator(
                lm, args.openai_api_key, args.openai_base_url
            )
        else:
            raise ValueError(f"Unknown LLM provider: {lm_provider}.")
        descs = description_generator.gen_facts_description(facts)
        for fact, desc in zip(facts, descs):
            if desc is None:
                desc = ["None", "None", "None", "None"]
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
