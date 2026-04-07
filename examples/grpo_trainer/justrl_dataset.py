from __future__ import annotations

import datasets
import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


class JustRLRLDataset(RLHFDataset):
    SUFFIX_PROMPT = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            if data_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=data_file)["train"]
            elif data_file.endswith(".json") or data_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.dataframe.map(
            self._append_justrl_suffix,
            num_proc=self.num_workers,
            desc="Adding JustRL suffix prompt",
        )
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _append_justrl_suffix(self, doc: dict):
        messages = doc[self.prompt_key]
        if not messages:
            return doc

        first_message = dict(messages[0])
        if not first_message.get("content", "").endswith(self.SUFFIX_PROMPT):
            first_message["content"] = f"{first_message['content']}{self.SUFFIX_PROMPT}"

        updated_messages = list(messages)
        updated_messages[0] = first_message

        updated_doc = dict(doc)
        updated_doc[self.prompt_key] = updated_messages
        return updated_doc