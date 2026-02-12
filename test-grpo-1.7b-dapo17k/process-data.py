from pathlib import Path

import numpy as np
import pandas as pd
from rich.pretty import pprint
from tqdm import tqdm


def apply_simple_prompt(prompt) -> list[dict[str, str]]:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    elif isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list):
        raise TypeError(f"Expected prompt to be a list/ndarray, got {type(prompt)!r}")
    if not prompt or not isinstance(prompt[0], dict):
        raise TypeError(f"Expected prompt to be a non-empty list of dicts, got {prompt!r}")

    content = prompt[0]["content"]
    start = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer (without quotes) "
        "where $Answer is the answer to the problem.\n\n"
    )
    end = '\n\nRemember to put your answer on its own line after "Answer:".'
    assert start in content and end in content, f"Unexpected prompt format {content}."
    content = content.replace(start, "").replace(end, "")
    content = f"{content.strip()}" + r" Please reason step by step, and put your final answer within \boxed{}."
    prompt[0]["content"] = content
    return prompt


def process(path: Path):
    data = pd.read_parquet(path)
    print(f"Loaded {len(data)} records from {path}")
    print(data.iloc[0])

    first_prompt = data.iloc[0]["prompt"]
    print("Before processing prompt:")
    pprint(first_prompt)

    tqdm.pandas()
    data["prompt"] = data["prompt"].progress_apply(apply_simple_prompt)

    print("After processing prompt:")
    pprint(data.iloc[0]["prompt"])
    out_path = path.parent / f"{path.stem}-processed.parquet"
    data.to_parquet(out_path)
    print(f"Saved processed data to {out_path}")


if __name__ == "__main__":
    process(Path("/root/verl/data/aime-2024.parquet"))
    process(Path("/root/verl/data/dapo-math-17k.parquet"))
