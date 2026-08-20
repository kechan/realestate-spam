# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`realestate-spam` (package name `realestate_spam`) is a research/notebook-driven library for real estate lead message classification and related NLP tasks: spam/lead-quality classification, LLM-based data augmentation, LocalLogic neighborhood-profile rewriting via LLM, embeddings, language detection, sentiment, toxicity, and zero-shot classification.

This is not an application with a CLI entrypoint — it's a library consumed from Jupyter notebooks (see the `unit_test_display_names_df` feather fixture and `EditableDataFrame` ipywidgets helper, both notebook-oriented). Code should be written to be imported and driven interactively rather than run as scripts.

## Development commands

Install in editable mode:
```
pip install -e .
```

Run all tests:
```
pytest
```

Run a single test file / test function:
```
pytest tests/test_balanced_sampler.py
pytest tests/test_balanced_sampler.py::test_balanced_sampler
```

There is no lint/format tooling configured in this repo.

## Architecture

The package is organized by pipeline stage under `realestate_spam/`:

- **`data/`** — dataset construction and exploration
  - `preprocessing.py` — cleans raw `NOTE`/`DISPLAY_NAME` message data (`keep_user_input_only`, `keep_user_input_str_only`, `cleanup_spaces_colons` strip system-generated boilerplate like "Website Contact Request - ", MLS IDs, listing IDs, URLs). `PreprocessingPipeline` (ABC) → `StandardDataPipeline` and `CleanseSimpleDisplayNameDataPipeline` turn a raw dataframe into a `text`/`label` training frame (`text` = `"DISPLAY_NAME is <name>; <note>"` or `CONTACT_STRUCT_INFO|NOTE`). `BalancedSampler` produces a class-balanced sample from a grouped dataframe.
  - `augment.py` — LLM-based data augmentation. `AugGPT` (ABC-ish base using legacy `openai.ChatCompletion.create`) with subclasses `TestAugGPT`, `KvCoreAugGPT`, `ChineseAugGPT`, each defining its own prompt template and response-parsing/repair logic (`_format_chatgpt_responses`, `_partial_json_parse`) because chatGPT responses are sometimes malformed/truncated JSON. `AugmentedDataGenerator` (ABC-ish base) and subclasses (`TestAugmentedDataGenerator`, `KvCoreAugmentedDataGenerator`, `KaggleToxicAugmentedDataGenerator`, `ChineseAugmentedDataGenerator`) splice augmented/external text into a source message at positions matched by a shared list of system-boilerplate regexes (mirrors the cleanup patterns in `preprocessing.py`), to synthesize new labeled `TEST`/`NOT_SPAM`/`SPAM` training rows.
  - `disillation.py` (sic — filename is misspelled, keep as-is for import compatibility) — dataset distillation for creating smaller representative subsets (e.g., for few-shot prompts or fast iteration) via t-SNE + KMeans (`KMeansSamplingDataDistiller`) or t-SNE + grid sampling (`UniformGridSamplingDataDistiller`), both implementing the `DataDistiller` ABC (`dim_reduce` → `generate_clusters`/grid → `distill`).
  - `search.py` — `VectorSimSearch`: cosine-similarity nearest-neighbor search over a precomputed embedding column, either by query string (needs `embedding_model`) or by existing dataframe index.
  - `visualization.py` — `InteractiveScatter`: plotly-based 2D scatter (e.g. of t-SNE output) colored by `class_label`, with support for overlaying centroids/selected points.

- **`llm/chatgpt.py`** (576 lines) — higher-level LLM "agents", each wrapping the `raa` package's `RAA` completion class (`self.raa = RAA(llm_model=..., sys_prompt=self.system_prompt, api_key_env_var=...)`):
  - `DisplayNameOriginAgent` (L13) — infers whether a display name looks like a real person's name and its likely origin.
  - `FirstTouchNoteSpamDetectorAgent` (L64) — the core spam/lead classifier prompt (classifies into `NOT_SPAM`/`SPAM`/`TEST`/`NOT_SURE` using a guideline list, optional n-shot examples); includes `recover_json` for salvaging malformed LLM JSON output.
  - `DataAugmentationAgent` (L200) — a newer/alternate augmentation agent (JSON-keyed-by-input-message) distinct from `data/augment.py`'s `AugGPT` family.
  - `LocalLogicGPTRewriter` (L252) — rewrites neighborhood-profile sections (`housing`, `transport`, `services`, `character`) with property-type/transaction-type-specific tone and RAG-style stat injection; parses `<section>...</section>` tagged LLM output. Handles both `openai<1.0` and `openai>=1.3.0` exception shapes when unwrapping `tenacity` `RetryError`s. System prompt built in `_construct_sys_prompt()`, user prompt (with RAG stats / placeholder guidelines / the "always start with X" opening-sentence rule) built in `construct_openai_user_prompt()`; call `print_prompt(**kwargs)` to dump the exact system+user text that gets sent to the LLM, for debugging. This is the class `realestate_content_transformer`'s pipeline imports for city/property-type page rewrites — see that repo's CLAUDE.md.

- **`models/`** — thin wrappers around HuggingFace/sentence-transformers models for inference only (`.eval()`, `torch.no_grad()`), each with batched `device`-aware embedding/prediction methods and a device-appropriate cache-clearing function (`torch.cuda.empty_cache` / `torch.mps.empty_cache` / no-op for CPU): `embedding.py` (`EmbeddingModel`, `InstructorEmbeddingModel`), `language_detect.py` (`EfficientLanguageDetector` — cascades from fast `gcld3`/`cld3`/`langdetect` to the heavier `papluca/xlm-roberta-base-language-detection` model only when the fast path is low-confidence), `sentiment.py`, `summarize.py` (T5-family), `toxic.py`, `zero_shot_classification.py` (NLI-based zero-shot via entailment hypotheses).

- **`utils/`** — `misc.py` (token counting via `tiktoken`, string truncation, `flatten_list`, `html_unescape`, `show_overlap` for comparing dataframes), `model.py` (`run_model_evaluation`/`run_model_inference` — TensorFlow/Keras batched inference helpers, support single model or ensemble averaging), `ui.py` (`EditableDataFrame`, an ipywidgets-based inline-editable table for labeling in a notebook; monkey-patches `pd.DataFrame.render`).

## Notable conventions

- Dataframes throughout use uppercase column names from the source system: `NOTE` (the message body), `DISPLAY_NAME`, `TOUCH_ENTRY_ID`, `CONTACT_STRUCT_INFO`, `class_label` (`NOT_SPAM`/`SPAM`/`TEST`), plus a derived lowercase `text`/`label` pair once a `PreprocessingPipeline` has run.
- Code relies on `.q(...)` (pandas query shorthand) and `.defrag_index(...)` methods on `pd.DataFrame` — these are monkey-patched in externally by the `realestate_core` package (see `utils/ui.py`'s import of `realestate_core.common.utils`), not defined in this repo. Expect `AttributeError` if that package isn't imported/installed in the environment.
- `llm/chatgpt.py` depends on an external `raa` package (`RAA` class) for actual OpenAI completion calls; `data/augment.py` instead calls the legacy `openai.ChatCompletion.create` API directly — these are two independent, not-yet-unified approaches to LLM calls in the codebase.
- Several regex-based "strip system-generated boilerplate" expression lists (MLS IDs, "Website ... Request -", listing IDs, URLs) are duplicated verbatim across `data/preprocessing.py` and multiple generator classes in `data/augment.py` — if you fix/extend one, check whether the others need the same fix.

## External dependencies (siblings, not in this repo)

None of these are installed from PyPI — all are local editable installs (`pip install -e .`), so `grep`/`find` across this repo alone will never find their source.

- **`raa`** — the `RAA` chat-completion wrapper class used throughout `llm/chatgpt.py`. Source: `/Users/kechan/Developer/raa` (note: NOT under Google Drive like the other sibling repos — it's a plain local path). GitHub: `kechan/raa`. Core class in `raa/raa.py` (`RAA.__init__`, `get_completion`, `get_completion_from_messages`). Reads the OpenAI key from the env var named by `api_key_env_var` (defaults to `OPENAI_API_KEY`; this repo's agents typically pass `JUMPTOOLS_OPENAI_API_KEY` explicitly).
- **`realestate_core`** — provides `.q()`/`.defrag_index()` DataFrame monkey-patches and other shared utils. Sibling repo under the same `Developer/` Google Drive folder.
- **Reverse dependency**: `realestate_content_transformer` (sibling repo, same `Developer/` folder) imports `LocalLogicGPTRewriter` from this repo's `llm/chatgpt.py` for its production content-rewrite pipeline — changes to that class's prompt construction or `rewrite()` return shape affect that repo's `pipeline.py` directly.

## Rollback Reference — Pre-v1 (city-level median metrics) Commit

`realestate_content_transformer`'s "v1" feature (dated median-price metrics on city-level
pages) required a `construct_openai_user_prompt()` guideline change in this repo's
`llm/chatgpt.py`. If v1 ever needs a "nuclear" rollback (redeploy old code + rerun), this
is the last commit on `origin/main` **before** that change landed:

- `realestate_spam` @ `6079342830d3c50185ce575cb0c1bbeef60027b2` — "subpage prompt change to put stats in the 1st sentence"
- (paired repo: `realestate_content_transformer` @ `628cf8996cf3a26c6fd4d678e81efab6bc883961` — see that repo's CLAUDE.md)

Note this only reverts *code* — it does not undo any city-level ES content already
published to PROD. For that, see `realestate_content_transformer`'s
`scripts/rollback_city_level_v1_median_metrics.py`, which restores archived pre-v1
content directly, without needing a code rollback at all — the preferred path; the
git-commit rollback above is the fallback of last resort.
