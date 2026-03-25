# Contextual Bandit Implementation TODO

## Current Status

The file `simulator/policy/contextual_bandit_stub.py` contains a `ContextualBanditPolicy` class that has the correct public interface but uses a simplified gradient-descent weight update instead of true LinUCB linear algebra.

The stub correctly:
- Defines feature vectors of dimension 5
- Implements UCB score computation with diagonal covariance approximation
- Provides `select()` and `update()` methods with correct signatures
- Computes exploration bonus proportional to `alpha * sqrt(uncertainty)`

---

## LinUCB Algorithm Description

LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm that:

1. Represents each arm (comment candidate) as a feature vector **x** ∈ ℝᵈ
2. Maintains a parameter estimate **θ** = A⁻¹**b** where:
   - **A** = XᵀX + λI (regularized covariance matrix)
   - **b** = Σ rₜ · **xₜ** (reward-weighted feature sum)
3. Computes UCB score: score(**x**) = **θ**ᵀ**x** + α √(**x**ᵀA⁻¹**x**)
4. Selects the arm with the highest UCB score
5. Updates A and **b** after observing reward r: A ← A + **x****x**ᵀ, **b** ← **b** + r**x**

The exploration term α √(**x**ᵀA⁻¹**x**) quantifies uncertainty: it is large when the feature space near **x** has been under-explored.

### Disjoint vs. Shared Models

- **Shared LinUCB**: One (A, **b**) pair for all arms → efficient, assumes arms share the same reward structure
- **Disjoint LinUCB**: Separate (Aₐ, **bₐ**) per arm → more flexible, requires tracking per-arm history

In our setting, "arms" are comment candidates that change every turn, so a **shared model** is appropriate.

---

## Feature Vector Design

Current feature vector (dimension d=5):

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | sentiment_normalized | (sentiment + 1) / 2, range [0, 1] |
| 1 | novelty_score | range [0, 1] |
| 2 | safety | 1 - toxicity_score, range [0, 1] |
| 3 | question_flag | 1.0 if question, 0.0 otherwise |
| 4 | bias | always 1.0 |

### Potential Extensions

- Add persona_group one-hot encoding (8 groups → +8 dims)
- Add turn_normalized (turn / num_turns) for temporal awareness
- Add affinity_to_streamer from viewer state
- Add topic match score: does comment topic match streamer's current topic?

---

## Reward Signal Options

The reward signal r ∈ [0, 1] determines what behavior the bandit optimizes for.

| Option | Formula | Optimizes for |
|--------|---------|---------------|
| Engagement | 1 if viewer commented next turn after being ignored | Viewer retention |
| Sentiment | (avg_sentiment_next_turn - avg_sentiment_this_turn + 1) / 2 | Community mood |
| Safety | 1 - toxicity of next toxic comment | Community safety |
| Composite | 0.3*engagement + 0.3*sentiment + 0.2*safety + 0.2*novelty | Balanced |

Currently the stub uses a composite reward computed at selection time (without lookahead).
A proper implementation would compute reward at the **next turn** based on observed outcomes.

---

## Implementation Steps

### Step 1: Fix A matrix and Sherman-Morrison update

```python
import numpy as np

class ContextualBanditPolicy(SelectionPolicy):
    def __init__(self, alpha: float = 1.0, lambda_: float = 1.0):
        self.alpha = alpha
        d = 5  # feature dimension
        self.A = lambda_ * np.eye(d)       # d x d matrix
        self.A_inv = (1/lambda_) * np.eye(d)  # maintain inverse directly
        self.b = np.zeros(d)               # d-dim vector
        self.theta = np.zeros(d)           # A_inv @ b

    def _ucb_score(self, x: np.ndarray) -> float:
        exploit = self.theta @ x
        variance = x @ self.A_inv @ x
        return exploit + self.alpha * np.sqrt(max(variance, 0))

    def update(self, selected, reward: float):
        x = np.array(_extract_features(selected))
        # Sherman-Morrison rank-1 update: A_inv <- A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        Ax = self.A_inv @ x
        denom = 1.0 + x @ Ax
        self.A_inv -= np.outer(Ax, Ax) / denom
        self.b += reward * x
        self.theta = self.A_inv @ self.b
```

### Step 2: Add delayed reward computation

Reward should be computed at turn t+1 based on observed viewer behavior:
- Did the selected viewer comment again? (+engagement)
- Did average sentiment improve? (+mood)
- Were fewer toxic comments seen? (+safety)

### Step 3: Add warm-start from ScoreBasedPolicy

For the first 10-20 turns, fall back to ScoreBasedPolicy until the bandit has sufficient data. This avoids cold-start issues.

### Step 4: Hyperparameter search

- alpha: controls exploration/exploitation tradeoff. Try [0.1, 0.5, 1.0, 2.0]
- lambda_: regularization strength. Try [0.01, 0.1, 1.0]
- Use compare experiments to find optimal settings

---

## References

1. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). **A contextual-bandit approach to personalized news article recommendation.** WWW 2010.
2. Lattimore, T., & Szepesvári, C. (2020). **Bandit Algorithms.** Cambridge University Press.
3. Sherman-Morrison formula: https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
4. Nemotron-Personas-Japan dataset: to be linked when available
