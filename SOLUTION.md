# SMILES 2026 Hallucination Detection Solution

## Run

```bash
pip install -r requirements.txt
python solution.py
```

Verified locally with Python 3.12.10, PyTorch 2.11.0+cu128, CUDA, and an RTX 4090.

## Changes

* changed `aggregation.py`, `probe.py`, and `splitting.py`
* fixed files `solution.py`, `evaluate.py`, and `model.py` are unchanged

## Method

The solution uses frozen Qwen2.5-0.5B hidden states. `solution.py` feeds the full prompt plus response, so `aggregation.py` recovers the response boundary by tokenizing prompts from `data/dataset.csv` and `data/test.csv` in the official extraction order.

The final aggregation mode is `max_l12_l13`: response-token max pooling from layer 12 concatenated with response-token max pooling from layer 13. This gives a 1792-dimensional feature vector. If the response span cannot be recovered safely, the fallback preserves the same feature dimension by using the last real token.

The probe is `StandardScaler` plus `LogisticRegression`. The regularization strength is selected inside `fit()` from `CANDIDATE_C = (0.003, 0.01, 0.03, 0.1)` using out-of-fold predictions. The final model is a 5-model bootstrap ensemble with averaged probabilities, and the decision threshold is also tuned inside `fit()`.

## Validation

Validation uses 5-fold stratified validation on labelled `dataset.csv`. The metrics are held-out fold metrics. I do not report accuracy for unlabelled `data/test.csv`.

## Results

| Metric | Value |
|---|---:|
| Majority baseline accuracy | 0.7010 |
| Held-out accuracy | 0.7590 |
| Held-out F1 | 0.8431 |
| Held-out AUROC | 0.7911 |
| Feature dimension | 1792 |
| Number of folds | 5 |

## Notes

There is no Qwen fine-tuning. I did not keep an MLP as the final probe because the labelled dataset is small. I tested a few response-only aggregation variants locally and kept the layer 12 plus layer 13 max-pooling variant for its balance of quality and simplicity.

## Output

`python solution.py` creates:

* `results.json`
* `predictions.csv`

`predictions.csv` format: `id,label`
