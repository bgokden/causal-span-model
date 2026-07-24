"""Push a trained causal span model to the Hugging Face Hub.

Uploads the checkpoint directory as-is: ``config.json`` + tokenizer + weights
(``model.safetensors``) + ``onnx/model.onnx`` + the model card (``README.md``).
Requires an HF token (``huggingface-cli login`` or ``HF_TOKEN``).

This PUBLISHES. Defaults to a PRIVATE repo; pass ``--public`` to make it public.
Confirm the repo id and visibility before running.
"""

import argparse
import os
import sys

# Files not worth uploading (intermediate training state).
_IGNORE = ["checkpoint-*/*", "training_args.bin"]


def push(
    model_dir: str,
    repo_id: str,
    private: bool = True,
    token: str | None = None,
    commit_message: str = "Add causal span model (mDeBERTa-v3 BIO tagger)",
) -> str:
    from huggingface_hub import HfApi

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=_IGNORE,
    )
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("repo_id", help="e.g. berkgokden/causal-span-mdeberta")
    parser.add_argument("--public", action="store_true",
                        help="make the repo public (default: private)")
    parser.add_argument("--token", default=None, help="HF token (else uses cached login)")
    args = parser.parse_args(argv)

    url = push(args.model_dir, args.repo_id, private=not args.public, token=args.token)
    print(f"pushed to {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
