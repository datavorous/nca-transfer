"""Convert a Hugging Face text dataset to the uint16 files used by OpenWebTextDataset."""

import argparse
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


def write_tokens(rows, text_column: str, output: Path, max_samples: int | None) -> None:
    encoder = tiktoken.get_encoding("gpt2")
    with output.open("wb") as stream:
        for index, row in enumerate(rows):
            if max_samples is not None and index >= max_samples:
                break
            tokens = encoder.encode_ordinary(row[text_column])
            tokens.append(encoder.eot_token)
            np.asarray(tokens, dtype=np.uint16).tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gair-prox/open-web-math-pro")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.005)
    parser.add_argument("--max-train-samples", type=int)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    dataset = load_dataset(args.dataset, split=args.split)
    splits = dataset.train_test_split(test_size=args.validation_fraction, seed=42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tokens(splits["train"], args.text_column, args.output_dir / "train.bin", args.max_train_samples)
    write_tokens(splits["test"], args.text_column, args.output_dir / "val.bin", None)


if __name__ == "__main__":
    main()
