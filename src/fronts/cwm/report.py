"""Quality reporting for a synthesised model, in the paper's Table 1 shape.

Why the train/test/online split is the whole point: the GAP between train
and test is the diagnostic. Train accuracy says the model fit the data it
was shown; test accuracy says whether it learned the dynamics or memorised
the recordings; online accuracy -- measured during play, on states its own
policy visited -- says whether it survives contact with a planner that
trusts it. A model that scores 0.95 on train and 0.60 on test has not
almost worked; it has overfit, and the fix is different from the fix for a
model that scores 0.60 everywhere.

The paper's own Gin rummy result is the honest failure case a report must
be able to show: 0.78 train, 0.75 test transition accuracy, 500 LLM calls,
budget exhausted -- and the agent built on that model lost badly to a
ground-truth opponent. A reporting layer that cannot display a number that
bad -- that averages the splits, or omits the call count -- is hiding the
information an operator needs to decide NOT to deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Split", "ModelQualityReport"]


Split = str
TRAIN: Split = "train"
TEST: Split = "test"
ONLINE: Split = "online"


@dataclass(slots=True)
class ModelQualityReport:
    """Transition and inference accuracy for train, test and online
    separately, satisfying the ``ModelReport`` protocol.

    The protocol's ``transition_accuracy`` / ``inference_accuracy``
    properties expose the TEST values: the held-out split is the honest
    headline, and a caller who wants the full picture calls
    ``format_table()``.
    """

    transition_accuracy_train: float = 0.0
    transition_accuracy_test: float = 0.0
    transition_accuracy_online: float = 0.0
    # None means NOT MEASURED, rendered as "n/a". Scoring inference needs a
    # sampler to score (see ``fronts.cwm.inference.inference_accuracy``);
    # copying the transition numbers into these columns would manufacture a
    # measurement that never happened.
    inference_accuracy_train: float | None = None
    inference_accuracy_test: float | None = None
    inference_accuracy_online: float | None = None
    llm_calls: int = 0
    passed_tests: int = 0
    total_tests: int = 0

    @property
    def transition_accuracy(self) -> float:
        return self.transition_accuracy_test

    @property
    def inference_accuracy(self) -> float | None:
        return self.inference_accuracy_test

    def format_table(self) -> str:
        """A plain-text table in the shape of the paper's Table 1.

        Accuracy as a two-decimal fraction, one row per split, so the train
        and test columns sit next to each other and the gap between them is
        the first thing the eye finds."""
        def cell(value: float | None) -> str:
            return f"{value:>12.2f}" if value is not None else f"{'n/a':>12}"

        header = f"{'split':<8} {'transition':>12} {'inference':>12}"
        rows = [
            f"{TRAIN:<8} {self.transition_accuracy_train:>12.2f} "
            f"{cell(self.inference_accuracy_train)}",
            f"{TEST:<8} {self.transition_accuracy_test:>12.2f} "
            f"{cell(self.inference_accuracy_test)}",
            f"{ONLINE:<8} {self.transition_accuracy_online:>12.2f} "
            f"{cell(self.inference_accuracy_online)}",
        ]
        footer = (
            f"tests: {self.passed_tests}/{self.total_tests} passed, "
            f"llm calls: {self.llm_calls}"
        )
        return "\n".join([header, *rows, footer])
