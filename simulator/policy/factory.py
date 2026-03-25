"""Policy factory for creating selection policies by name."""
from __future__ import annotations

from .contextual_bandit_stub import ContextualBanditPolicy
from .random_policy import RandomPolicy
from .rule_based import RuleBasedPolicy
from .score_based import ScoreBasedPolicy

POLICIES = {
    "random": RandomPolicy,
    "rule_based": RuleBasedPolicy,
    "score_based": ScoreBasedPolicy,
    "contextual_bandit": ContextualBanditPolicy,
}


def create_policy(name: str, **kwargs):
    """Create a policy instance by name.

    Args:
        name: Policy name (random, rule_based, score_based, contextual_bandit).
        **kwargs: Additional arguments passed to the policy constructor.

    Returns:
        An instance of the requested policy.

    Raises:
        ValueError: If the policy name is unknown.
    """
    if name not in POLICIES:
        raise ValueError(
            f"Unknown policy: {name}. Available: {list(POLICIES.keys())}"
        )
    return POLICIES[name](**kwargs)
