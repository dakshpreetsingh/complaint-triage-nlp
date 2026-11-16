# Complaint Triage: Baseline vs. Transformer

**Live demo:** https://complaint-triage-nlp-3vkejvzbpnjbxlmhupkkwv.streamlit.app/

Classifies a customer complaint into one of 11 financial products (credit card, mortgage, student loan, ...) so it can be routed to the right team. The project compares a simple TF-IDF + Logistic Regression baseline with a fine-tuned DistilBERT, and asks whether the transformer is worth its cost.

## Result in one paragraph

On a locked test set of 7,730 complaints, the tuned baseline (macro-F1 **0.659**) and fine-tuned DistilBERT (**0.666**) are statistically indistinguishable: the 95% interval for the difference (DistilBERT minus baseline) is **-0.015 to +0.028**, which includes zero. DistilBERT is about **290x slower per complaint on CPU** and **18x larger**, so the app serves the baseline.

## Problem

A support team receives complaints as free text. Routing each one to the right product team by hand is slow, so the goal is to predict the product from the text alone. The label is the product the consumer selected when filing, so it is a noisy proxy for what a complaint is "really" about.

## Data

- **Source:** CFPB Consumer Complaint Database (the Consumer Financial Protection Bureau is the original source), loaded from the Parquet mirror `Mouwiya/cfpb-consumer-complaints` on Hugging Face (about 16.6 million rows).
- **Filtering:** kept complaints that have a narrative and were received on or after 2024-01-01. Product names changed over the years, so a recent window gives one consistent label set. Drew a random sample of 60,000.
- **Cleaning (60,000 to 40,428 rows):** removed redaction tokens (`XXXX`, `XX/XX/XXXX`), dropped complaints under 10 words, dropped texts that appear under more than one product label, dropped exact duplicates (28.7% of the raw sample), and dropped products with fewer than 100 complaints (none needed dropping).

| Product | Complaints | Share |
|---|---|---|
| Credit reporting or other personal consumer reports | 25,984 | 64.3% |
| Debt collection | 4,482 | 11.1% |
| Checking or savings account | 2,659 | 6.6% |
| Credit card | 2,547 | 6.3% |
| Money transfer, virtual currency, or money service | 1,727 | 4.3% |
| Mortgage | 873 | 2.2% |
| Vehicle loan or lease | 696 | 1.7% |
| Student loan | 631 | 1.6% |
| Payday loan, title loan, personal loan, or advance loan | 443 | 1.1% |
| Prepaid card | 248 | 0.6% |
| Debt or credit management | 138 | 0.3% |

The data is heavily imbalanced (largest class is about 188x the smallest), so the main metric is **macro-F1**. Always predicting the majority class scores 0.071.

## Method

1. **Group-aware split.** Many complaints are near-copies of templated letters. A random split would put look-alikes in both train and test and inflate scores. I used `StratifiedGroupKFold` with groups defined by the first 100 characters (lowercased), giving train 26,220 / validation 6,478 / test 7,730, and verified that no group appears in more than one split.
2. **Baseline.** TF-IDF (word unigrams and bigrams, `min_df=3`, sublinear TF) + Logistic Regression (`C=1`, `class_weight="balanced"`). Chosen from 8 settings on the validation set (validation macro-F1 0.669).
3. **DistilBERT.** `distilbert-base-uncased` fine-tuned in a plain PyTorch loop: max length 256 tokens, batch size 32, AdamW (lr 3e-5, weight decay 0.01), 6% warmup then linear decay, mixed precision, gradient clipping, and class weights tempered with a square root (full balanced weights span about 188x and destabilise training). Best epoch kept by validation macro-F1 (epochs 1 to 3: 0.614, 0.642, 0.676; about 8.5 minutes on a free Colab GPU).
4. **Evaluation.** Both models scored once on the locked test set. Uncertainty comes from a bootstrap that resamples near-duplicate groups (1,000 resamples), so copies of one letter do not count as independent evidence.

## Results (test set)

| | Baseline | DistilBERT |
|---|---|---|
| Macro-F1, all 11 products | 0.659 | 0.666 |
| Macro-F1, without the 29-example class | 0.713 | 0.720 |
| 95% interval, macro-F1 | 0.636 to 0.682 | 0.642 to 0.688 |
| Difference (DistilBERT minus baseline) | | -0.015 to +0.028 (DistilBERT ahead in 74% of resamples) |
| Time per complaint | 0.88 ms (CPU) | 2.34 ms (GPU), 255 ms (CPU) |
| File size | 15 MB | 268 MB |

Timings are rough: measured on Colab's shared hardware, DistilBERT CPU time on 200 complaints, tokenisation excluded.

Per-product F1 on the test set:

| Product | n (test) | Baseline | DistilBERT |
|---|---|---|---|
| Checking or savings account | 526 | 0.78 | 0.80 |
| Credit card | 516 | 0.69 | 0.75 |
| Credit reporting or other personal consumer reports | 4,891 | 0.90 | 0.92 |
| Debt collection | 935 | 0.69 | 0.71 |
| Debt or credit management | 29 | 0.12 | 0.12 |
| Money transfer, virtual currency, or money service | 269 | 0.74 | 0.75 |
| Mortgage | 168 | 0.83 | 0.87 |
| Payday loan, title loan, personal loan, or advance loan | 102 | 0.52 | 0.47 |
| Prepaid card | 51 | 0.63 | 0.54 |
| Student loan | 118 | 0.72 | 0.77 |
| Vehicle loan or lease | 125 | 0.63 | 0.62 |

DistilBERT's gains were consistent across validation and test only on the larger classes (credit reporting, debt collection, credit card, mortgage). Differences on classes with 29 to 126 test examples changed direction between validation and test and should be treated as noise.

## What the errors look like

The most frequent baseline mistakes on the validation set were credit reporting complaints predicted as debt collection (335), credit card (138), vehicle loan (48) and student loan (41), and money transfer and checking account confused with each other (47 and 32). Reading examples of the hardest class ("Debt or credit management") showed complaints that were plainly about credit reports, collections or checking-account fees. My hypothesis is that much of the remaining error comes from overlapping labels rather than model weakness. This is supported by the confusion pairs and the examples, but I have not tested it directly.

## Limitations

- Each model was trained once, so the bootstrap reflects test-set variation, not training randomness.
- The smallest classes have 29 to 51 test examples, so their scores are noisy.
- Near-duplicate grouping uses the first 100 characters, a pragmatic proxy that will miss some copies and group some unrelated texts.
- DistilBERT's validation score was still rising at epoch 3; longer training was not tried.
- The model must choose one of 11 products and has no "none of these" option, and it only knows US financial complaints from 2024 onward. Out-of-domain text still gets a confident-looking answer.
- App scores rank products; they are not calibrated probabilities (class weights inflate scores for small classes).
- Labels are the product the consumer selected, which is not always what the text is about.

## The app

The Streamlit app serves the baseline: paste a complaint and it shows the top 3 products plus the words that pushed toward the top prediction (for a linear model, the contribution of each word is its TF-IDF value times the learned weight). Example: "my loan's EMI was deducted twice from my credit card" ranks Credit card first (0.51), then Student loan (0.15) and Payday/personal loan (0.12). That is a defensible but ambiguous call, since the label depends on which loan it is, and "EMI" is probably absent from the US training data.

## Repository

```
app.py              Streamlit app
baseline.joblib     Trained TF-IDF + Logistic Regression
requirements.txt    Pinned dependencies
notebooks/          Full analysis notebook (data prep, baseline, DistilBERT, comparison)
```

The DistilBERT weights (268 MB) are not included.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the analysis, open the notebook in Google Colab with a GPU runtime. All random seeds are 42.

## Credits

Data: Consumer Financial Protection Bureau, Consumer Complaint Database.
Tools: Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, Streamlit.
