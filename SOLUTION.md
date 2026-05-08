# SMILES 2026 Hallucination Detection Solution

## Run

From the repository root:

```bash
pip install -r requirements.txt
python solution.py
```

My run used Python 3.12.10, PyTorch 2.11.0+cu128, CUDA, and an RTX 4090.

The solution only changes:

- `aggregation.py`
- `probe.py`
- `splitting.py`

The original `solution.py`, `evaluate.py`, and `model.py` are unchanged.

## Approach

I use Qwen2.5-0.5B only as a frozen feature extractor.

The baseline last-token representation felt too narrow, so I switched to response-only features. Since `aggregate()` does not receive token ids or prompt boundaries, I recover prompt lengths inside `aggregation.py` by tokenizing the prompts from `data/dataset.csv` and `data/test.csv` in the same order as the official extraction loop.

For each sample, I take layer 13 and max-pool hidden states over the response span. If the response span cannot be recovered safely, the code falls back to the last real token.

This gives one 896-dimensional vector per sample.

The probe is deliberately simple:

- `StandardScaler`
- `LogisticRegression(C=0.01)`
- 5 bootstrap resamples
- averaged probabilities

The threshold is selected inside `fit()` using out-of-fold predictions. I do it there because the final prediction path calls `fit()` and then immediately writes `predictions.csv`.

## Validation

I use 5-fold stratified validation on the labelled `dataset.csv`. The reported numbers below are held-out fold metrics. I do not report accuracy for the final `data/test.csv`, since its labels are not public.

## Results

| Metric | Value |
|---|---:|
| Majority baseline accuracy | 0.7010 |
| Held-out accuracy | 0.7489 |
| Held-out F1 | 0.8329 |
| Held-out AUROC | 0.7869 |
| Feature dimension | 896 |
| Number of folds | 5 |

## What I did not keep

I did not use the MLP probe from the starter code. With only 689 labelled examples, it is easy to overfit.

I also avoided a larger multi-layer PCA setup in the final version. It added complexity, while the response-only 896-dimensional probe was simpler and gave a solid validation score.

## Output

Running `python solution.py` creates:

- `results.json`
- `predictions.csv`

`predictions.csv` has the required `id,label` format.
