# Decision: Confidence Scoring Penalties

**Status:** Under review
**Date:** (removed)
**Context:** Gateway service confidence calculation

## Summary

The gateway service applies penalties and weighted scoring to the confidence value returned by the `/identify` endpoint. The scoring logic lives in `services/gateway/src/gateway/routers/identify.py` (`calculate_confidence()`) with configuration in `services/gateway/config.yaml` under the `scoring:` section.

## Current Behavior

| Scenario | Formula | Effect |
|---|---|---|
| Geometric score > 0.5 | `0.6 * similarity + 0.4 * geometric_score` | Weighted blend |
| Geometric score <= 0.5 | `0.3 * similarity + 0.2 * geometric_score` | Heavy penalty (max 0.5) |
| Geometric enabled but no score | `similarity * 0.7` | 30% penalty |
| Geometric disabled | `similarity * 0.85` | 15% penalty |

## Origin

This scoring system was introduced in commit `85b7d41` ("feat(gateway): add API routers for all endpoints"). It was copied verbatim from the implementation prompt at `docs/implementation-prompts/gateway-service-implementation-task.md`, which prescribed these exact formulas and magic numbers.

The devlog (`docs/devlog/gateway-service-implementation.md`) retroactively explains the rationale as "Confidence penalty signals reduced reliability," but this was written by the same agent that implemented the code — it is describing its own output, not citing external requirements.

## Evidence

- No ticket, issue, or requirement requested this scoring system.
- No empirical evaluation supports the specific weight values (0.6, 0.4, 0.3, 0.2, 0.7, 0.85).
- No A/B test or accuracy benchmark compared these penalties against raw similarity scores.
- The values were invented in the implementation prompt with no stated justification.

## Risk

The penalties artificially deflate confidence scores, which could cause valid matches to fall below `confidence_threshold` and be discarded. The `geometric_missing_penalty` (0.7) is particularly aggressive — a perfect similarity score of 1.0 gets reduced to 0.7 simply because geometric verification didn't run.

## Tuning Plan

The evaluation dataset at `data/evaluation/` contains reference images (`objects/`), visitor test photos (`pictures/`), and ground-truth labels (`labels.csv`). This data can be used to empirically determine whether the penalties help or hurt, and to find optimal values.

### Step 1: Establish Baseline

Run the evaluation pipeline with current config and record accuracy metrics:

```bash
just evaluate
```

This produces a report in `reports/evaluation/` with per-image match results, accuracy, and timing.

### Step 2: Test Without Penalties

Set all scoring weights to pass-through (confidence = raw similarity):

```yaml
scoring:
  geometric_score_threshold: 0.0
  geometric_high_similarity_weight: 1.0
  geometric_high_score_weight: 0.0
  geometric_low_similarity_weight: 1.0
  geometric_low_score_weight: 0.0
  geometric_missing_penalty: 1.0
  embedding_only_penalty: 1.0
```

Re-run `just evaluate` and compare accuracy against baseline.

### Step 3: Grid Search Over Penalty Values

If penalties do improve accuracy, sweep over values to find optimal ones:

- `geometric_missing_penalty`: test `[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`
- `embedding_only_penalty`: test `[0.7, 0.8, 0.85, 0.9, 0.95, 1.0]`
- `geometric_score_threshold`: test `[0.3, 0.4, 0.5, 0.6, 0.7]`
- For the weighted blends, test different `(similarity_weight, score_weight)` pairs that sum to 1.0

For each configuration:
1. Update `services/gateway/config.yaml`
2. Restart the gateway service
3. Run `just evaluate`
4. Record accuracy, precision, and false-positive rate

### Step 4: Validate on Held-Out Data

Split the evaluation dataset (e.g., 70/30) so tuning is not done on the same data used for final accuracy reporting. Tune on the 70% split, validate on the 30% split.

### Step 5: Document Results

Record the winning configuration with its accuracy numbers in this document and update `config.yaml` accordingly.
