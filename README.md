# Pipeline for classifier model


CLI-first workflow for cleaning, curating, and fine-tuning a DistilBERT classifier on the [`morgul10/booking_reviews`](https://huggingface.co/datasets/morgul10/booking_reviews) dataset. The pipeline bundles data ingest, cleaning, splitting, data quality checks (Cleanlab), training, evaluation, behavioral tests (Giskard), and explanations (Captum) into a Typer-based command suite. The goal is to fine-tune DistilBERT, or any other text model compatible with `AutoModelForSequenceClassification`, with an additional classification head so that it recognises positive and negative booking reviews.

### Quick Run
Prerequisites (before cloning):
- Git
- [uv](https://github.com/astral-sh/uv) (manages Python 3.11 env + deps)
- Optional: CUDA-capable GPU if you plan to train on `device="cuda"` (without a GPU the pipeline falls back to CPU)

Then run:

```bash
# 1. Clone this repo and enter it
git clone https://github.com/Po33ski/Booking-Sentiment.git
cd Booking-Sentiment

# 2. Install Python 3.11 env + dependencies (incl. dev tools: jupyter, pytest, ruff) via uv
uv sync

# 3. Run the entire pipeline with the quick config
uv run booking-sentiment all --config configs/quick.json

# 4. Try the trained model interactively
uv run booking-sentiment inference --config configs/quick.json
```

### Jupyter Notebook

The notebook presents the entire pipeline in a simple way, without the extra CLI / classes etc.
It includes slightly fewer features than the full CLI pipeline, but also a few steps that were added specifically for the notebook.
The notebook is easy to run if you have suitable hardware: you can start it locally after running `uv sync` and selecting the uv environment as the kernel,
or (often more simply) you can open it directly in Google Colab: [`https://colab.research.google.com/`](https://colab.research.google.com/).

To rerun individual CLI stages instead of the full pipeline, replace the final command with the specific subcommand (e.g., `load`, `clean`, `split`, `quality`, `tune`, `evaluate`, `inference`, etc.). See more in the Running the Pipeline section.

### Processing Steps
1. **Data hygiene** – drop boilerplate texts (e.g. "No Negative"), empty strings and duplicates. Texts that appear with both labels (once as a positive and once as a negative review) are removed, because their label is ambiguous.
2. **Splitting** – draw a stratified sample of `dataset.sample_size` rows and split it into train/validation/test subsets (default 0.6/0.1/0.3, editable in the JSON config).
3. **Cleanlab quality pass (train + validation only)** – convert texts to embeddings with `all-MiniLM-L6-v2`, fit a logistic regression (`sklearn.linear_model.LogisticRegression`, regularized via `quality.regularization_c`) with cross-validation (`quality.cv_folds`), and use the out-of-fold probabilities to find label issues, near duplicates and outliers. The demo fixes them automatically: flagged labels are replaced with the predicted label, one example per near-duplicate group is kept (the whole group is dropped if its labels disagree), and outliers are dropped. Manual review is encouraged in production.
   - The **test split is never modified**, so the evaluation measures agreement with the original labels.
   - Train/validation rows that are almost identical to a test row (cosine similarity ≥ `quality.test_leak_similarity`) are dropped to avoid leakage.
4. **Fine-tuning** – fine-tune `distilbert/distilbert-base-uncased` plus a classification head via Hugging Face `Trainer`. The model is evaluated on the validation split after every epoch and the checkpoint with the best validation **MCC** is kept. Intermediate checkpoints are deleted after training, so `artifacts/finetuned_model/` contains only the final model and tokenizer.
5. **Evaluation** – compute Precision, Recall, F1, AUROC and MCC on the test split; see the Metrics section for details.
6. **Behavioral testing** – leverage [Giskard](https://github.com/Giskard-AI/giskard) to probe robustness, unfairness, and sensitivity through slicing and input perturbations. Giskard calls a prediction function that runs the fine-tuned model on raw text and returns the probabilities of both classes.
7. **Explainability** – compute Captum **Input × Gradient** attributions for a few test samples. The attributions are computed with respect to the word embeddings (passed to the model via `inputs_embeds`), so this works for any supported model, not only DistilBERT. Results are saved as TSV files and as colored HTML visualizations. For more explanation see the Interpretability Example section.
8. **Inference** – run the saved fine-tuned model interactively. You can enter a review, see how it is classified, and exit the loop any time by typing `quit` or `q`. This step is not part of `all`, because it waits for user input.

## Features
- **Data ingest & preview** – pull raw Booking.com positives/negatives from Hugging Face and persist quick-look parquet artefacts.
- **Cleaning** – configurable term filtering, length constraints, lowercasing, case-insensitive deduplication and removal of texts with conflicting labels.
- **Deterministic splitting** – stratified sampling and train/valid/test splits driven by `DatasetConfig` and `SplitConfig`.
- **Quality analysis** – SentenceTransformer + logistic regression to detect label issues, near duplicates and outliers via Cleanlab, with automatic fixes on train/valid and a test-leakage check.
- **HF fine-tuning** – DistilBERT (or any HF checkpoint) trained via `Trainer`, with best-checkpoint selection by validation MCC.
- **Evaluation** – Precision/Recall/F1/AUROC/MCC on the untouched test split, written to `metrics.json`.
- **Explainability & scans** – Captum Input × Gradient token attributions and a Giskard behavioral scan.
- **Interactive inference** – reuse the fine-tuned checkpoint to classify ad-hoc reviews from the CLI and inspect probabilities.
- **Strict configuration** – unknown or misspelled keys in the JSON config raise an error instead of being silently ignored.

## Models

By default the pipeline fine‑tunes `distilbert/distilbert-base-uncased`, but it is designed to work with **any Hugging Face checkpoint that is compatible with `AutoModelForSequenceClassification` for text**.

- **Supported model families (examples)**
  - BERT: `bert-base-uncased`, `bert-base-multilingual-cased`, `distilbert-base-uncased`
  - RoBERTa: `roberta-base`, `distilroberta-base`
  - XLM-R: `xlm-roberta-base`
  - Other encoder-style text models that work with `AutoModelForSequenceClassification`

- **How it works**
  The fine‑tuning code always loads the model via:
  `AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)`.
  For any text encoder supported by this API, Transformers automatically attaches a 2‑class classification head, so you can switch architectures by changing only `fine_tune.model_name` in the JSON config.

- **What is not supported out of the box**
  - Non‑text models (e.g. ViT, CLIP, Whisper, image/audio models).
  - Seq2seq generative models (e.g. T5, BART) unless you adapt the code to use `AutoModelForSeq2SeqLM` and a different training/evaluation loop.


## Metrics

- **Precision**: when the model predicts the positive class, precision measures how often this prediction is actually correct. High precision means a low fraction of false positives (FP).
- **Recall**: measures how much of the truly positive class the model manages to detect. High recall means a low fraction of false negatives (FN).
- **F1-score**: the harmonic mean of precision and recall, combining both metrics into a single number.
  Because it is a harmonic mean, F1 heavily penalizes low values – to obtain a high F1, **both** precision and recall must be high.
- **Area Under Receiver Operating Characteristic (AUROC / ROC AUC)**: less common in NLP, but with several useful properties. It takes predicted probabilities into account and evaluates the model across many thresholds (how high the positive-class probability must be to call a sample positive). For each threshold we compute the fractions of true positives and false positives and then aggregate them. High AUROC means the model assigns high probabilities to correct classes and avoids false positives even for low thresholds.
- **Matthews Correlation Coefficient (MCC)**: can be seen as an analogue of Pearson’s correlation coefficient for binary variables. It has favourable statistical properties and can reveal model failures even when accuracy or AUROC look high. MCC on the validation split is also used to pick the best checkpoint during fine-tuning.


## Interpretability Example
During the `explain` step we compute Captum Input × Gradient attributions to see which tokens push the model toward its prediction. Each TSV under `artifacts/explain/` contains the original text plus per‑token attribution scores, and a matching `sample_N.html` file shows the same tokens colored by attribution. An illustrative snapshot:

```
# true_label  0
# pred_label  0
# pred_proba  0.997457
# text        the maid come in one morning when the do not disturb sign was on the door
token    attr
the      0.1359
maid     0.3670
come     0.1031
in      -0.2122
...
not     -0.6537
disturb -0.2533
...
door    -0.0950
```

- `pred_label` is the explained class (by default the class predicted by the model) and `pred_proba` is the model's probability of that class.
- `attr` is the Input × Gradient attribution (normalized to unit length); positive numbers push the prediction toward the explained class, negative numbers push it away.
- Large magnitude = stronger influence.
- We strip special tokens (e.g. `[CLS]`/`[SEP]`) before exporting, so only human-readable tokens remain.

These files make it easy to inspect why the model believed a review was positive/negative and to spot spurious cues.

## Project Layout
```
booking_sentiment/   ← package with CLI, configs, training, explainability etc.
configs/             ← JSON configs (e.g., `quick.json`)
tests/               ← pytest tests (run with `uv run pytest`)
artifacts/           ← generated datasets, models, explanations, metrics (gitignored)
pyproject.toml       ← uv metadata (Python 3.11, deps like transformers, cleanlab)
```

## Running the Pipeline
Each command accepts `--config PATH` (defaults to built-ins). Example using `configs/quick.json`:

```bash
# Full pipeline (load → clean → split → quality → tune → evaluate → scan → explain)
uv run booking-sentiment all --config configs/quick.json

# Single step
uv run booking-sentiment load --config configs/quick.json

# Ad-hoc inference (after training finishes)
uv run booking-sentiment inference --config configs/quick.json

# Delete all artifacts (asks for confirmation; add --yes to skip it)
uv run booking-sentiment purge --config configs/quick.json
```

Core commands (executed in this order during `all`):
1. `load` – download dataset and save raw/neg/pos previews.
2. `clean` – output `artifacts/clean.parquet`.
3. `split` – stratified sampling + train/valid/test split → `artifacts/splits_raw/`.
4. `quality` – Cleanlab on train+valid (label fixes, near-duplicate and outlier removal, test-leakage check) → final splits in `artifacts/splits/` (test copied unchanged) + report in `artifacts/quality_logs/`.
5. `tune` – fine-tune the model, save the best one to `artifacts/finetuned_model/`.
6. `evaluate` – compute Precision/Recall/F1/AUROC/MCC on the test split and write `metrics.json`.
7. `scan` – run the Giskard behavioral scan.
8. `explain` – Captum Input × Gradient for sampled test reviews → TSV + HTML files.

`inference` is run separately: it prompts for any review text, runs the saved model, and displays whether it is predicted as positive or negative together with the positive-class probability.

## Docker Usage
The image is meant for running the CLI with an already trained model: it contains the runtime dependencies (without dev tools such as Jupyter), `configs/`, and `artifacts/finetuned_model/` (train the model first). Build it from the repo root:

```bash
docker build -t booking-sentiment-cli .
```

The entrypoint exposes the Typer CLI and by default starts interactive inference:

```bash
# interactive inference with the bundled model
docker run --rm -it booking-sentiment-cli

# show CLI help
docker run --rm -it booking-sentiment-cli --help
```

To run other steps (or the full pipeline) in the container, mount a persistent directory to `/app/artifacts` so that models, parquet splits, and other outputs have enough disk space and survive container restarts. Training on a GPU inside Docker additionally requires the NVIDIA Container Toolkit and `--gpus all`:

```bash
docker run --rm -it --gpus all -v "$(pwd)/artifacts:/app/artifacts" booking-sentiment-cli all --config configs/quick.json
```

In managed environments (Cloud Run, ECS, etc.) attach a persistent volume or sufficiently large disk to `/app/artifacts` to avoid running out of space during training.


## Configuration Highlights (`configs/quick.json`)
- `dataset.sample_size` – number of rows sampled (stratified) from the cleaned data before splitting (set `null` to use all rows).
- `cleaning` – `remove_terms` (texts equal to one of these are dropped), `min_text_len`, `lowercase_text`, `casefold_dedup`.
- `split` – `train_frac`/`valid_frac`/`test_frac` must each be > 0 and sum to 1; `stratify` preserves class balance.
- `fine_tune` – `model_name`, `learning_rate` (default `2e-5`), `epochs`, `batch_size`, `warmup_ratio`, `weight_decay`, `seed`, `device` (`cuda` or `cpu`).
- `quality` – `embedding_model_name`, `cv_folds`, `regularization_c`, `iterations` (number of Cleanlab passes), `test_leak_similarity`, `device`.
- `paths.artifacts_dir` – default `artifacts/`.

Unknown keys are rejected, so a typo in the config fails fast with a clear error.

## Outputs
- `artifacts/raw_preview.parquet`, `neg_preview.parquet`, `pos_preview.parquet`, `clean.parquet`
- `artifacts/splits_raw/{train,valid,test}.parquet` – splits before the quality step
- `artifacts/splits/{train,valid,test}.parquet` – final splits used for training and evaluation
- `artifacts/quality_logs/quality_log.md` – Cleanlab report (which rows were fixed/dropped)
- `artifacts/finetuned_model/` (model + tokenizer + `preprocessing.json`, which tells inference to lowercase inputs when the training texts were lowercased)
- `artifacts/metrics.json`
- `artifacts/explain/` (per-sample TSV attributions + HTML visualizations)
- `artifacts/giskard/giskard_scan.md` (Giskard scan report)

## Tests
```bash
uv run pytest
```
The tests use a tiny randomly initialised model, so they run in a few seconds without a GPU or internet access.

## Notes
- Use `uv run booking-sentiment --help` to list commands/options.
- Cleanlab logs show how many labels were fixed and which rows were dropped; review `quality_log.md` if you prefer manual curation.
- After modifying configs or cleaning rules, rerun from the relevant step onward (e.g., `split → quality → tune → evaluate`). Running `split` removes the previous final splits, so `quality` has to be run again before `tune`.
- Training metrics are also logged to MLflow in `./mlruns` (MLflow is installed as a Giskard dependency and the Hugging Face `Trainer` picks it up automatically). Set `DISABLE_MLFLOW_INTEGRATION=TRUE` to turn this off.
