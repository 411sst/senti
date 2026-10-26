"""Agent #2: the NEWS2 tool-agent.

Wraps the deterministic NEWS2 scorer as an :class:`~agents.base.Agent` so the
clinical baseline speaks the same protocol as the learned agents. Two
consequences worth being explicit about:

* **Nothing is trained.** NEWS2 is a fixed published spec (CLAUDE.md rule #6:
  it is a baseline, not something to improve), so :meth:`_fit_model` is a
  genuine no-op. It still goes through the base class's isotonic-on-val
  calibration, which is what makes its emitted risk directly comparable to the
  learned agents' -- a raw NEWS2 of 7 means nothing as a probability until it
  is mapped onto observed event rates.
* **The latent is the breakdown, not the total.** Instead of a learned
  embedding, the agent emits the vector of seven NEWS2 sub-scores
  (respiratory rate, SpO2, temperature, systolic BP, heart rate, oxygen,
  consciousness). The orchestrator therefore sees *which* parameters are
  driving the score -- "this patient is deteriorating on respiratory rate
  specifically" -- which a single 0-20 number throws away.
"""

import numpy as np

from agents.base import KEY_COLUMNS, Agent
from mimic_cohort import NEWS2_COMPONENTS, news2_component_scores

# NEWS2's maximum attainable score; used only to express the raw 0-20 score on
# a [0, 1] scale for reporting raw ECE (calibration itself is rank-based and
# does not care about the scale).
NEWS2_MAX = 20.0


class News2Agent(Agent):
    """Deterministic NEWS2 as an agent: 7 sub-scores + a calibrated risk."""

    name = "news2"
    latent_dim = len(NEWS2_COMPONENTS)
    component_names = tuple(NEWS2_COMPONENTS)

    def _fit_model(self, train_df, val_df):
        """No-op: NEWS2 is a fixed spec with no parameters to learn.

        The base class still calibrates on val after this returns.
        """
        return self

    def _emit_raw(self, df):
        """``(keys, component sub-scores, NEWS2 total)`` by ``(stay_id, hour)``."""
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        components, _, _ = news2_component_scores(ordered)
        raw = components.sum(axis=1)                 # the 0-20 NEWS2 score
        return (ordered[list(KEY_COLUMNS)].copy(),
                components.astype(np.float32),
                raw)

    def raw_probability(self, df):
        """Raw score rescaled to [0, 1], for reporting an honest raw ECE.

        The NEWS2 score is not a probability, so ECE against it is undefined
        without a mapping. Dividing by the maximum score is the crudest
        possible one -- which is exactly the point: it shows how badly an
        uncalibrated NEWS2 misstates risk, and how much the isotonic step buys.
        """
        return self.raw_risk(df) / NEWS2_MAX
