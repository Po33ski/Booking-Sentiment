# Pipeline for classifier model 


CLI-first workflow for cleaning, curating, and fine-tuning a DistilBERT classifier on the [`morgul10/booking_reviews`](https://huggingface.co/datasets/morgul10/booking_reviews) dataset. The pipeline bundles data ingest, quality checks (Cleanlab), splitting, training, evaluation, explanation, and Giskard tests into a Typer-based command suite. The goal is to fine-tune DistilBERT or any other model (which is compatible with AutoModelForSequenceClassification for text) from with an additional classification head so that it recognise positive and negative booking reviews.

### Quick Run
Prerequisites (before cloning):
- Git
- [uv](https://github.com/astral-sh/uv) (manages Python 3.11 env + deps)
- Optional: CUDA-capable GPU if you plan to train on `device="cuda"`

Then run:

```bash
# 1. Clone this repo and enter it
git clone https://github.com/Po33ski/Booking-Sentiment.git
cd Booking-sentiment

# 2. Install Python 3.11 env + dependencies via uv
uv sync

# 3. Run the entire pipeline with the quick config
uv run booking-sentiment all --config configs/quick.json
```
### Jupyter Notebook

The notebook presents the entire pipeline in a simple way, without the extra CLI / classes ect.  
It includes slightly fewer features than the full CLI pipeline, but also a few steps that were added specifically for the notebook.  
The notebook is easy to run if you have suitable hardware: you can start it locally after running `uv sync` and selecting the UV environment as the kernel,  
or (often more simply) you can open it directly in Google Colab: [`https://colab.research.google.com/`](https://colab.research.google.com/).

To rerun individual CLI stages instead of the full pipeline, replace the final command with the specific subcommand (e.g., `load`, `clean`, `train`, `evaluate`, `inference`, etc.). See more in the Running the Pipeline section.

### Processing Steps
1. **Data hygiene** – drop duplicate rows and empty strings with Pandas safeguards.
2. **Cleanlab quality pass** – convert texts to embeddings with `all-MiniLM-L6-v2`, fit a logistic regression (`sklearn.linear_model.LogisticRegression`, regularized via `cfg.regularization_c`) using cross-validation (`cv_folds`), and use the resulting probabilities to flag label issues, outliers, and near duplicates. The demo applies automatic relabeling/deduplication, though manual review is encouraged in production.
3. **Splitting** – create train/validation/test subsets (default 0.6/0.1/0.3, editable in the JSON config) with optional stratification and sampling.
4. **Fine-tuning** – fine-tune `distilbert/distilbert-base-uncased` plus a classification head via Hugging Face `Trainer`. The default run uses 5 epochs (adjust based on `sample_size`). Evaluation metrics are computed immediately afterward; see the Metrics section for details.
5. **Behavioral testing** – leverage [Giskard](https://github.com/Giskard-AI/giskard) to probe robustness, unfairness, and sensitivity through slicing and input perturbations. Because Giskard needs raw text, the embedding computation happens inside the prediction function, which reuses the logistic classifier trained above.
6. **Explainability** – run Captum Integrated Gradients on tokenized samples. The `configure_interpretable_embedding_layer()` hook swaps in Captum’s embedding tracker, while a thin wrapper passes `attention_mask` through the model. We then visualize token attributions. For more explanation see the Interpretability Example section.
7. **Inference** - run the saved fine-tuned model for inference. User can enter a command and see how it will be classified, and exit the loop any time by typing `quit` or `q`.

## Features
- **Data ingest & preview** – pull raw Booking.com positives/negatives from Hugging Face and persist quick-look parquet artefacts.
- **Cleaning** – configurable term filtering, length constraints, case folding, and deduplication.
- **Quality analysis** – SentenceTransformer + logistic regression to detect label issues/outliers via Cleanlab, with optional automatic fixes.
- **Deterministic splitting** – stratified train/valid/test splits driven by `SplitConfig`.
- **HF fine-tuning** – DistilBERT (or any HF checkpoint) trained via `Trainer`, with MCC tracking and saved tokenizer/model/tokenized datasets.
- **Evaluation & debugging** – metrics plus top False Positives/Negatives and most uncertain predictions.
- **Explainability & scans** – Captum-based token attributions and optional Giskard safety scan.
- **Interactive inference** – reuse the fine-tuned checkpoint to classify ad-hoc reviews from the CLI and inspect probabilities.

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
  For any text encoder supported by this API, Transformers automatically attaches a 2‑class classification head, so you can switch architectures by changing only `tune.model_name` in the JSON config.

- **What is not supported out of the box**  
  - Non‑text models (e.g. ViT, CLIP, Whisper, image/audio models).  
  - Seq2seq generative models (e.g. T5, BART) unless you adapt the code to use `AutoModelForSeq2SeqLM` and a different training/evaluation loop.


## Metrics
Precision: if the model predicts the positive class, precision measures how sure we can be that it's really the positive class. High precision means low fraction of false positives (FP).
Recall: measures how much of the positive class the model detected. High recall means low fraction of false negatives (FN).
F1-score: a harmonic mean of precision and recall, aggregates them into one number for convenience. 
Harmonic mean heavily penalizes small numbers, so to get high value, both precision and recall have to be high, not just one of those. 
Area Under Receiver Operating Characteristic (AUROC / ROC AUC): is less frequently used in NLP, but has a few beneficial properties. It takes into consideration model probability predictions. For different thresholds (percentage above which we assume positive class) we measure the fractions of true positives and false positives, and aggregate those numbers. To achieve high AUROC, the model has to predict the right class with high probability, and avoid false positives even for low thresholds.
Matthews Correlation Coefficient (MCC): can be thought of as Pearson correlation, but for binary variables. It has favorable statistical properties, and can spot model failures even when accuracy or AUROC are high. 

## MLFlow:
The project is instrumented with Hugging Face's built‑in MLflow integration. When you run fine‑tuning, the `Trainer` automatically starts an MLflow run and logs hyperparameters, metrics (including MCC), and useful metadata. By default, these runs are stored locally under the `mlruns/` directory, which acts as MLflow's file‑based tracking backend. You can point MLflow to a different tracking URI if you want to use a central tracking server instead of the local `mlruns/` folder.

## Interpretability Example
During the `explain` step we run Captum Integrated Gradients to see which tokens push the model toward a positive vs. negative prediction. Each TSV under `artifacts/explain/` contains the original text plus per-token attribution scores. A snapshot:

```
# true_label  0
# pred_label  0
# pred_proba  0.002543
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

- `attr` is the attribution value from Integrated Gradients; positive numbers nudge the prediction toward the positive class, negative toward the negative class.
- Large magnitude = stronger influence. In this sample words like “not”/“disturb” carry negative attributions, which aligns with the true label (0 = negative review).
- We strip special `[CLS]/[SEP]` tokens before exporting, so only human-readable tokens remain.

These files make it easy to inspect why the model believed a review was positive/negative and to spot spurious cues.

## Project Layout
```
booking_sentiment/   ← package with CLI, configs, training, explainability etc.
configs/             ← JSON configs (e.g., `quick.json`)
artifacts/           ← generated datasets, models, explanations, metrics (gitignored)
pyproject.toml       ← uv/poetry metadata (Python 3.11, deps like transformers, cleanlab)
```

## Running the Pipeline
Each command accepts `--config PATH` (defaults to built-ins). Example using `configs/quick.json`:

```bash
# Full pipeline
uv run booking-sentiment all --config configs/quick.json

# Single step
uv run booking-sentiment load --config configs/quick.json

# Ad-hoc inference (after training finishes)
uv run booking-sentiment inference --config configs/quick.json

```

## Docker Usage
Build the container from the repo root when you need an isolated environment (local machine, server, or cloud runner):

```bash
docker build -t booking-sentiment-cli .
```

The entrypoint exposes the Typer CLI. Mount a persistent directory to `/app/artifacts` so that models, parquet splits, and other outputs have enough disk space and survive container restarts:

```bash
# show CLI help
docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" booking-sentiment-cli --help

# run the full pipeline
docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" booking-sentiment-cli all --config configs/quick.json
```

In managed environments (Cloud Run, ECS, etc.) attach a persistent volume or sufficiently large disk to `/app/artifacts` to avoid running out of space during training.

Core commands (executed in order during `all`):
1. `load` – download dataset and save raw/neg/pos previews.
2. `clean` – output `artifacts/clean.parquet`.
3. `quality` – run Cleanlab, optional label fixes, drop flagged outliers → `quality_fixed.parquet`.
4. `split` – create stratified parquet splits under `artifacts/splits/`.
5. `tune` – fine-tune DistilBERT, save to `artifacts/finetuned_model` and tokenized datasets.
6. `evaluate` – compute Precision/Recall/F1/AUROC/MCC and list difficult samples + write `metrics.json`.
7. `scan` – run Giskard safety scan.
8. `explain` – Captum Integrated Gradients for sampled test reviews → TSV files.
9. `inference` – prompt for any review text, run the saved model, and display whether it is predicted as positive or negative together with the positive-class probability.


## Configuration Highlights (`configs/quick.json`)
- `dataset.sample_size` – subsample before quality/split (set `null` for full set).
- `cleaning` – remove boilerplate terms, enforce min length, case folding.
- `split` – fractions must sum to 1; `stratify` preserves class balance.
- `tune` – `model_name`, `learning_rate`, `epochs`, `device`.
- `quality` – `embedding_model_name`, `cv_folds`, `regularization_c`, plus thresholds `label_issue_threshold` / `max_label_fixes` for safer automatic relabeling.
- `paths.artifacts_dir` – default `artifacts/`.

## Outputs
- `artifacts/raw_*.parquet`, `clean.parquet`, `quality_fixed.parquet`
- `artifacts/splits/{train,valid,test}.parquet`
- `artifacts/finetuned_model/` (model + tokenizer) and `artifacts/tokenized/`
- `artifacts/metrics.json`
- `artifacts/explain/` (per-sample TSV attributions)
- `artifacts/giskard/` (if `scan` executed)


## Notes
- Use `uv run booking-sentiment --help` to list commands/options.
- Cleanlab logs show how many labels were flagged/modified; tweak thresholds if you prefer manual review.
- After modifying configs or cleaning rules, rerun from the relevant step onward (e.g., `quality → split → train`).



