"""Agent #5b: the rich-notes agent -- a WINDOW of radiology reports, not just one.

The shipped notes agent (:mod:`agents.notes_agent`) embeds exactly one report:
the most recent one available before hour ``t``. That throws away everything
else in the record and, more importantly, gives the text no **temporal**
dimension -- the orchestrator's whole thesis is that trajectory matters, and a
single static embedding cannot express "the imaging picture is changing".

This agent aggregates every report available in a lookback window and exposes
three blocks:

``recent``  the most recent report's embedding -- what the old agent saw
``mean``    the mean embedding over the window -- the standing picture
``delta``   ``recent - mean(older reports)`` -- how the picture has MOVED

``delta`` is the point. It is zero when there is only one report, so any gain it
buys is attributable to having more than one.

**On note types.** MIMIC-IV-Note ships exactly two tables, ``discharge`` and
``radiology`` (verified: 331,793 and 2,321,355 rows; our own index contains only
``RR`` and ``AR``). Nursing and physician progress notes existed in MIMIC-III's
``NOTEEVENTS`` but were **not** carried into MIMIC-IV, so "richer clinical text"
here can only mean a richer *encoding* of radiology, never a richer source.
Discharge summaries remain excluded outright -- they are written a median of 57h
after ICU discharge and narrate the outcome being predicted.

**Leakage.** Identical discipline to the single-note agent, applied to every
note in the window rather than to one: a report counts only if
``0 < t - note_hour <= window``, where ``note_hour`` is
``GREATEST(storetime, charttime)`` -- availability, not acquisition. The strict
``> 0`` is what ``allow_exact_matches=False`` buys the single-note agent: a
report filed exactly at hour ``t`` is not yet readable at hour ``t``.

**Clinician suspicion (CLAUDE.md rule #4).** How MANY reports exist in a window
is a proxy for how much imaging was ordered, i.e. for how worried the team was.
That is a treatment-derived feature, so ``n_notes`` and ``span`` are **off by
default** and available only behind ``use_count_features``. Their standalone
discriminative power is reported by :func:`count_only_auprc` as a floor, the way
:func:`~agents.notes_agent.presence_only_auprc` is for presence.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import KEY_COLUMNS, Agent
from agents.note_encoder import EMBED_DIM
from agents.notes_agent import EMBED_CACHE, NOTE_INDEX, STALENESS_HOURS
from config import SEED
from metrics import auprc

LABEL = "label_24h"

# Which aggregation the agent uses. "recent" reproduces the information the
# shipped single-note agent sees, so it is the control arm for "does the window
# help?" rather than a straw man.
AGGREGATIONS = ("recent", "mean", "rich")


class WindowIndex:
    """Which notes each patient-hour can see, as a ragged index.

    Materialising a 768-dim vector per cohort row costs ~5 GB for the train
    split alone, so three blocks of them will not fit in memory. Instead we keep
    integer positions into the embedding table -- ~12 bytes per (row, note) pair
    -- and aggregate per batch. The embedding table itself is 127,542 x 768,
    about 390 MB, and is shared by every row.

    Within a row, ``positions`` is ordered most-recent-first, so element 0 is
    the report the single-note agent would have used.
    """

    def __init__(self, offsets, positions, ages, keys):
        self.offsets = offsets          # (N + 1,) int64
        self.positions = positions      # (M,) int32 -- rows of the embedding table
        self.ages = ages                # (M,) float32 -- hours before t
        self.keys = keys                # (N, 2) stay_id + hour

    def __len__(self):
        return len(self.offsets) - 1

    @property
    def counts(self):
        return np.diff(self.offsets)

    def subset(self, mask):
        """A view holding only the rows where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        counts = self.counts[mask]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        keep = np.concatenate(
            [np.arange(self.offsets[i], self.offsets[i + 1])
             for i in np.flatnonzero(mask)]) if mask.any() else np.array([], int)
        return WindowIndex(offsets, self.positions[keep], self.ages[keep],
                           self.keys[mask].reset_index(drop=True))


def build_window_index(cohort, note_index, note_ids, window=STALENESS_HOURS):
    """Resolve, for every cohort row, the notes available in its lookback window.

    A note is available at hour ``t`` when ``0 < t - note_hour <= window``. The
    strict inequality is the same rule the single-note agent gets from
    ``allow_exact_matches=False``: a report filed at hour ``t`` is not readable
    at hour ``t``.
    """
    lookup = {str(n): i for i, n in enumerate(note_ids)}

    left = cohort[["stay_id", "hour"]].copy()
    left["stay_id"] = left["stay_id"].astype("int64")
    left["hour"] = left["hour"].astype("float64")
    left = left.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    left["_row"] = np.arange(len(left), dtype=np.int64)

    right = note_index[["stay_id", "note_hour", "note_id"]].copy()
    right["stay_id"] = right["stay_id"].astype("int64")
    right["note_hour"] = right["note_hour"].astype("float64")
    right["_pos"] = [lookup.get(str(n), -1) for n in right["note_id"]]
    right = right[right["_pos"] >= 0]           # unencoded notes count as absent

    pairs = left.merge(right[["stay_id", "note_hour", "_pos"]], on="stay_id",
                       how="inner")
    age = pairs["hour"].to_numpy() - pairs["note_hour"].to_numpy()
    keep = (age > 0) & (age <= window)          # STRICTLY before t, within window
    pairs = pairs[keep]
    age = age[keep]

    # most-recent-first within a row, so element 0 is the single-note agent's pick
    order = np.lexsort((age, pairs["_row"].to_numpy()))
    rows = pairs["_row"].to_numpy()[order]
    positions = pairs["_pos"].to_numpy()[order].astype(np.int32)
    ages = age[order].astype(np.float32)

    counts = np.bincount(rows, minlength=len(left)).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return WindowIndex(offsets, positions, ages,
                       left[["stay_id", "hour"]].astype("int64"))


def aggregate_blocks(index, embeddings, rows, aggregation="rich"):
    """Per-row aggregated blocks + scalars for a batch of row indices.

    Returns ``(blocks (B, n_blocks * 768), has (B,), age (B,), n_notes (B,),
    span (B,))``. Rows with no note in the window get all-zero blocks; the head
    substitutes its learned "no report" vector there.
    """
    rows = np.asarray(rows)
    lo, hi = index.offsets[rows], index.offsets[rows + 1]
    counts = (hi - lo).astype(np.int64)
    covered = counts > 0

    n_blocks = 1 if aggregation in ("recent", "mean") else 3
    blocks = np.zeros((len(rows), n_blocks * EMBED_DIM), dtype=np.float32)
    has = covered.astype(np.float32)
    age = np.zeros(len(rows), dtype=np.float32)
    span = np.zeros(len(rows), dtype=np.float32)
    n_notes = counts.astype(np.float32)
    if not covered.any():
        return blocks, has, age, n_notes, span

    idx = np.flatnonzero(covered)
    age[idx] = index.ages[lo[idx]]                       # most recent
    span[idx] = index.ages[hi[idx] - 1]                  # oldest in window

    flat = np.concatenate([index.positions[a:b] for a, b in zip(lo[idx], hi[idx])])
    vectors = embeddings[flat]
    starts = np.concatenate([[0], np.cumsum(counts[idx])[:-1]]).astype(np.int64)
    sums = np.add.reduceat(vectors, starts, axis=0)
    means = sums / counts[idx][:, None]
    recent = vectors[starts]

    if aggregation == "recent":
        blocks[idx] = recent
    elif aggregation == "mean":
        blocks[idx] = means
    else:
        # delta = recent - mean(OLDER notes); exactly zero when there is only
        # one report, so any gain from delta comes from having more than one.
        older_sums = sums - recent
        older_counts = (counts[idx] - 1).astype(np.float32)
        safe = np.where(older_counts > 0, older_counts, 1.0)[:, None]
        older_means = np.where(older_counts[:, None] > 0, older_sums / safe, recent)
        delta = recent - older_means
        blocks[idx] = np.concatenate([recent, means, delta], axis=1)
    return blocks, has, age, n_notes, span


def count_only_auprc(y, n_notes):
    """AUPRC from the note COUNT alone -- the clinician-suspicion floor.

    How much imaging was ordered is a treatment decision, so this is the number
    that says how much of any gain could be explained by "sicker patients get
    more scans" rather than by what the scans say.
    """
    return auprc(y, n_notes)


class RichNoteHead(nn.Module):
    """Projects each 768-dim block, concatenates, then a small MLP.

    ``no_note_embedding`` is one learned vector per block, substituted wholesale
    on rows with no report. Every uncovered row therefore presents an identical
    input -- same vectors, scalars zeroed -- so the head is mathematically forced
    to emit one constant risk there. That keeps presence-separation auditable
    rather than letting it hide inside the embedding, exactly as in the
    single-note agent.
    """

    def __init__(self, n_blocks=3, embed_dim=EMBED_DIM, project=64, hidden=128,
                 latent=32, dropout=0.2, n_scalars=1):
        super().__init__()
        self.n_blocks, self.embed_dim = n_blocks, embed_dim
        self.no_note_embedding = nn.Parameter(torch.zeros(n_blocks * embed_dim))
        nn.init.normal_(self.no_note_embedding, std=0.02)
        self.project = nn.ModuleList(
            nn.Linear(embed_dim, project) for _ in range(n_blocks))
        self.body = nn.Sequential(
            nn.Linear(n_blocks * project + n_scalars, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.head = nn.Linear(latent, 1)
        self.latent_dim = latent

    def forward(self, blocks, has_note, scalars):
        mask = has_note.unsqueeze(-1)
        x = mask * blocks + (1.0 - mask) * self.no_note_embedding
        parts = [self.project[i](x[:, i * self.embed_dim:(i + 1) * self.embed_dim])
                 for i in range(self.n_blocks)]
        parts.append(scalars * mask)          # scalars are zero where uncovered
        latent = self.body(torch.cat(parts, dim=-1))
        return self.head(latent).squeeze(-1), latent


class RichNotesAgent(Agent):
    """Windowed radiology embeddings -> latent + calibrated risk."""

    name = "notes_rich"

    def __init__(self, aggregation="rich", window=STALENESS_HOURS, project=64,
                 hidden=128, latent=32, dropout=0.2, lr=1e-3, batch_size=4096,
                 max_epochs=15, patience=3, seed=SEED, embed_cache=EMBED_CACHE,
                 note_index=NOTE_INDEX, use_count_features=False, verbose=True):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        self.aggregation, self.window = aggregation, window
        self.hparams = dict(project=project, hidden=hidden, latent=latent,
                            dropout=dropout)
        self.lr, self.batch_size = lr, batch_size
        self.max_epochs, self.patience = max_epochs, patience
        self.seed, self.verbose = seed, verbose
        self.latent_dim = latent
        self.embed_cache, self.note_index_path = embed_cache, note_index
        # See the module docstring: note COUNT is a clinician-suspicion proxy.
        self.use_count_features = use_count_features
        self.net, self.stats, self._cache = None, None, None
        self.device = torch.device("cpu")

    @property
    def n_blocks(self):
        return 3 if self.aggregation == "rich" else 1

    @property
    def n_scalars(self):
        return 3 if self.use_count_features else 1

    def _load_cache(self):
        if self._cache is None:
            z = np.load(self.embed_cache, allow_pickle=True)
            self._cache = (z["embeddings"].astype(np.float32), z["note_ids"],
                           pd.read_parquet(self.note_index_path))
        return self._cache

    def _index_for(self, df):
        emb, ids, note_index = self._load_cache()
        return build_window_index(df, note_index, ids, self.window), emb

    def _batch(self, index, embeddings, rows):
        blocks, has, age, n_notes, span = aggregate_blocks(
            index, embeddings, rows, self.aggregation)
        if self.stats is not None:
            mean, std = self.stats
            blocks = np.where(has[:, None] > 0, (blocks - mean) / std, 0.0
                              ).astype(np.float32)
        scalars = ([age, n_notes, span] if self.use_count_features else [age])
        return blocks, has, np.stack(scalars, axis=1).astype(np.float32)

    def _labels(self, df, keys):
        ordered = df.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
        if not (ordered["stay_id"].to_numpy() == keys["stay_id"].to_numpy()).all():
            raise ValueError("window features are misaligned with the cohort rows")
        return ordered[LABEL].to_numpy(dtype=float)

    def _fit_stats(self, index, embeddings):
        """Standardise each block on COVERED train rows only."""
        sample = np.flatnonzero(index.counts > 0)
        if len(sample) > 200_000:
            sample = np.random.default_rng(self.seed).choice(
                sample, 200_000, replace=False)
        blocks, _, _, _, _ = aggregate_blocks(index, embeddings, sample,
                                              self.aggregation)
        mean = blocks.mean(axis=0) if len(blocks) else np.zeros(blocks.shape[1])
        std = blocks.std(axis=0) if len(blocks) else np.ones(blocks.shape[1])
        self.stats = (mean.astype(np.float32),
                      np.where(std > 1e-6, std, 1.0).astype(np.float32))

    def _fit_model(self, train_df, val_df):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_index, embeddings = self._index_for(train_df)
        val_index, _ = self._index_for(val_df)
        self.stats = None
        self._fit_stats(train_index, embeddings)

        ytr = self._labels(train_df, train_index.keys)
        yva = self._labels(val_df, val_index.keys)
        covered_va = val_index.counts > 0

        self.net = RichNoteHead(n_blocks=self.n_blocks, n_scalars=self.n_scalars,
                                **self.hparams).to(self.device)
        if self.verbose:
            n = sum(p.numel() for p in self.net.parameters())
            cov = float((train_index.counts > 0).mean())
            per = train_index.counts[train_index.counts > 0].mean()
            print(f"[{self.name}] {n:,} head params (encoder frozen); "
                  f"aggregation={self.aggregation} window={self.window:g}h; "
                  f"{100*cov:.1f}% of train hours covered, "
                  f"{per:.1f} reports per covered hour")

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        yt = torch.from_numpy(ytr.astype(np.float32))
        best, best_state, stale = -np.inf, None, 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = rng.permutation(len(train_index))
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                rows = order[i:i + self.batch_size]
                blocks, has, scalars = self._batch(train_index, embeddings, rows)
                logits, _ = self.net(torch.from_numpy(blocks),
                                     torch.from_numpy(has),
                                     torch.from_numpy(scalars))
                loss = F.binary_cross_entropy_with_logits(
                    logits, yt[torch.from_numpy(rows)])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(rows)

            scores = self._scores(val_index, embeddings)
            # Select on the NOTE-AVAILABLE subset: that is the honest verdict and
            # it cannot be gamed by learning who gets imaging ordered.
            sel = (auprc(yva[covered_va], scores[covered_va])
                   if yva[covered_va].sum() > 0 else auprc(yva, scores))
            if self.verbose:
                print(f"  epoch {epoch}: loss {total/len(order):.4f} | "
                      f"val AUPRC(note-available) {sel:.4f}")
            if sel > best:
                best, stale, best_state = sel, 0, {
                    k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    if self.verbose:
                        print(f"  early stop (no gain for {self.patience} epochs)")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        if self.verbose:
            print(f"[{self.name}] best val AUPRC (note-available) {best:.4f}")
        return self

    @torch.no_grad()
    def _forward(self, index, embeddings, want_latent=False):
        self.net.eval()
        logits, latents = [], []
        for i in range(0, len(index), self.batch_size):
            rows = np.arange(i, min(i + self.batch_size, len(index)))
            blocks, has, scalars = self._batch(index, embeddings, rows)
            lo, lat = self.net(torch.from_numpy(blocks), torch.from_numpy(has),
                               torch.from_numpy(scalars))
            logits.append(lo.numpy())
            if want_latent:
                latents.append(lat.numpy())
        return (np.concatenate(logits),
                np.concatenate(latents) if want_latent else None)

    def _scores(self, index, embeddings):
        logits, _ = self._forward(index, embeddings)
        return 1.0 / (1.0 + np.exp(-logits))

    def _emit_raw(self, df):
        if self.net is None:
            raise RuntimeError("agent is not fitted -- call fit() first")
        index, embeddings = self._index_for(df)
        logits, latents = self._forward(index, embeddings, want_latent=True)
        raw = 1.0 / (1.0 + np.exp(-logits))
        return index.keys, latents.astype(np.float32), raw

    def coverage(self, df):
        """``(has_note, n_notes)`` per row, in ``(stay_id, hour)`` order."""
        index, _ = self._index_for(df)
        counts = index.counts
        return counts > 0, counts.astype(float)
