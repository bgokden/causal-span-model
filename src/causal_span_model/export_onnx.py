"""Export a trained checkpoint to ONNX in the layout reasongraph consumes.

reasongraph's ``OnnxTokenClassifierExtractor`` expects, for a model directory
``<dir>``:

  * ``<dir>/config.json``     with ``id2label``  (written by training)
  * tokenizer files in ``<dir>``                 (written by training)
  * ``<dir>/onnx/model.onnx``  inputs ``input_ids`` + ``attention_mask``,
                               output per-token logits ``[seq_len, num_labels]``

optimum exports ``model.onnx`` plus copies of config/tokenizer into its output
directory, so exporting straight into ``<dir>/onnx`` satisfies the contract
(the root config/tokenizer from training stay in place). ``--check`` then runs
one inference through onnxruntime with the exact feeds the consumer uses, to
prove the export is loadable before it is shipped.
"""

import argparse
import json
import os
import sys


def export(model_dir: str, opset: int = 18) -> str:
    """Export ``model_dir`` to ``model_dir/onnx/model.onnx``. Returns the onnx path."""
    from optimum.exporters.onnx import main_export

    onnx_dir = os.path.join(model_dir, "onnx")
    main_export(
        model_name_or_path=model_dir,
        output=onnx_dir,
        task="token-classification",
        opset=opset,
    )
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"expected {onnx_path} after export; optimum wrote {os.listdir(onnx_dir)}"
        )
    return onnx_path


def verify(model_dir: str, text: str = "Heavy rainfall caused severe flooding.") -> None:
    """Run one inference with the consumer's exact feeds and check the contract."""
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    with open(os.path.join(model_dir, "config.json")) as handle:
        config = json.load(handle)
    if "id2label" not in config:
        raise ValueError("config.json is missing id2label; the consumer needs it")
    num_labels = len(config["id2label"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    onnx_path = os.path.join(model_dir, "onnx", "model.onnx")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_names = {i.name for i in session.get_inputs()}
    expected = {"input_ids", "attention_mask"}
    if not expected.issubset(input_names):
        raise ValueError(f"onnx inputs {input_names} do not cover {expected}")

    encoded = tokenizer(text, return_tensors="np")
    feeds = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    logits = session.run(None, feeds)[0]
    seq_len = feeds["input_ids"].shape[1]
    if logits.shape != (1, seq_len, num_labels):
        raise ValueError(
            f"logits shape {logits.shape} != expected (1, {seq_len}, {num_labels})"
        )
    print(f"verified: {onnx_path} loads and returns {logits.shape} logits")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="trained checkpoint directory")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true",
                        help="run one onnxruntime inference to verify the contract")
    args = parser.parse_args(argv)

    onnx_path = export(args.model_dir, opset=args.opset)
    print(f"exported {onnx_path}")
    if args.check:
        verify(args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
