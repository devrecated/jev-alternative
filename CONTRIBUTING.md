# Contributing

Thanks for helping out.

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Fork the repo, make a branch, and keep each pull request to one change.
3. Run `python validate_data.py data/*.jsonl` and `python -m py_compile *.py scripts/*.py` before you push. If you have a GPU, include the smoke run output (`--max-steps 60`) in the PR.
4. Never commit API keys, `.env` files, model weights, private datasets or results from your own runs. `.gitignore` covers the common ones.

By contributing you agree that your work is licensed under the [AGPL-3.0](LICENSE).
