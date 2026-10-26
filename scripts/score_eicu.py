"""Zero-shot external validation: score a MIMIC-trained stack on eICU-CRD.

    python scripts/score_eicu.py [--agents vitals_tcn news2 labs] [--epochs 8]

Nothing here is trained, calibrated or thresholded on eICU. Agents are fit on
MIMIC train and isotonic-calibrated on MIMIC val; the fusion is fit the same
way; eICU is then scored through the frozen result. eICU is a pure held-out
set in the strongest sense available -- a different set of 208 hospitals.

WHY THIS SCRIPT RE-FITS RATHER THAN LOADING. Only the orchestrator checkpoints
are persisted as weights; the per-modality agents cache *emissions*, not
models. Re-fitting on MIMIC with the audited seed reproduces them, and the
reproduction is checked against the shipped caches before any eICU number is
believed (--check-repro, on by default).

WHY IT WRITES TO ITS OWN EMBEDDING DIRECTORY. The shipped caches under
data/cache/embeddings/ are what the entire results corpus is paired against.
This script never writes there; it uses data/cache/embeddings_eicu/ so a
transfer experiment can never silently move a published number.

REPORTING NOTE. AUPRC is prevalence-dependent and the two cohorts differ
(4.74% vs 3.38% positive hours), so raw AUPRC is not comparable across them.
Every table below reports lift = AUPRC / prevalence alongside it, and AUROC,
which is prevalence-free.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines_mimic import load_splits  # noqa: E402
from config import SEED  # noqa: E402
from metrics import auprc, auroc, ece  # noqa: E402

LABEL = "label_24h"
VITALS = ["heart_rate", "resp_rate", "spo2", "sbp", "temperature"]
EICU_COHORT = REPO / "data" / "cache" / "eicu_cohort_hourly.parquet"
EICU_LABS = REPO / "data" / "cache" / "eicu_labs_hourly.parquet"
EICU_FLUIDS = REPO / "data" / "cache" / "eicu_fluids_hourly.parquet"
EICU_RESP = REPO / "data" / "cache" / "eicu_resp_hourly.parquet"
SHIPPED_EMB = REPO / "data" / "cache" / "embeddings"
EMB_DIR = REPO / "data" / "cache" / "embeddings_eicu"


def report_transfer(tag, ts, tr, es, er):
    """Pooled, matched-density and per-hospital tables for one scored config."""
    print(f"\n=== {tag} — pooled ===")
    row("MIMIC test", ts[LABEL].to_numpy(float), tr)
    row("eICU (208 hospitals)", es[LABEL].to_numpy(float), er)

    print(f"\n=== {tag} — matched charting density ===")
    for b in (12, 6, 3, 2):
        mm, me = fresh_mask(ts, b), fresh_mask(es, b)
        row(f"MIMIC  fresh <= {b}h", ts.loc[mm, LABEL].to_numpy(float), tr[mm])
        row(f"eICU   fresh <= {b}h", es.loc[me, LABEL].to_numpy(float), er[me])

    print(f"\n=== {tag} — per-hospital (>=20k hours, >=50 positive hours) ===")
    tmp = es[["hospitalid", LABEL]].copy()
    tmp["r"] = er
    rows = []
    for h, g in tmp.groupby("hospitalid"):
        y = g[LABEL].to_numpy(float)
        if len(g) >= 20000 and y.sum() >= 50:
            rows.append((auroc(y, g.r.to_numpy()),
                         auprc(y, g.r.to_numpy()) / y.mean()))
    d = pd.DataFrame(rows, columns=["auroc", "lift"])
    print(f"  {len(d)} hospitals | AUROC p10 {d.auroc.quantile(.1):.3f} "
          f"med {d.auroc.median():.3f} p90 {d.auroc.quantile(.9):.3f}")
    print(f"  {'':22} lift p10 {d.lift.quantile(.1):.2f} "
          f"med {d.lift.median():.2f} p90 {d.lift.quantile(.9):.2f}")


def two_head_scores(bank, splits, eicu, epochs, seed=42):
    """The SHIPPED FLAGSHIP: treatment head + mortality head, noisy-OR (T9).

    Reproduces scripts/two_head.py's construction against an arbitrary bank:
    a label-overridden orchestrator per component (train AND val overridden so
    early stopping selects on that component), each isotonic-calibrated on the
    V_fit half of val and plateau-strictified, then combined as
    ``s = 1 - (1 - cT)(1 - cD)``.

    Calibration matters here in a way it does not for a single head: noisy-OR
    combines two *probabilities*, so the calibrators affect the combined
    ranking, not just its scale. Both are fitted on MIMIC val and applied
    unchanged to eICU -- which is exactly the deployment question.
    """
    from decompose import treatment_only_label
    from distill_lead import DistilledOrchestrator, make_soft_frame
    from fusion_common import FusionContext
    from fusion_ladder import _strictify
    from sklearn.isotonic import IsotonicRegression

    ctx = FusionContext(splits)
    train_o = splits["train"].sort_values(["stay_id", "hour"]).reset_index(drop=True)
    val_o = ctx.val_ordered
    targets = {
        "T": (treatment_only_label(train_o), treatment_only_label(val_o)),
        "D": (train_o["label_24h_death"].to_numpy(float),
              val_o["label_24h_death"].to_numpy(float)),
    }

    heads = {}
    for name, (y_tr, y_va) in targets.items():
        t0 = time.time()
        soft = pd.concat([make_soft_frame(train_o, y_tr),
                          make_soft_frame(val_o, y_va)])
        head = DistilledOrchestrator(bank=bank, max_epochs=epochs, seed=seed,
                                     verbose=False, soft_frame=soft)
        head._fit_model(splits["train"], splits["val"])
        raw_val = head.raw_risk(splits["val"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[ctx.fit_mask], y_va[ctx.fit_mask])
        out = {}
        for key, df in (("test", splits["test"]), ("eicu", eicu)):
            if df is None:
                continue
            raw = head.raw_risk(df)
            # _strictify adds 1e-9 * rank to break isotonic plateaus, which
            # pushes a calibrated 1.0 fractionally above 1. Harmless for the
            # ranking it exists to fix, but these are probabilities and the
            # noisy-OR below multiplies them, so clip back into [0, 1].
            out[key] = np.clip(_strictify(iso.transform(raw), raw), 0.0, 1.0)
        heads[name] = out
        print(f"[two-head] {name}-head fit + scored in {time.time()-t0:.0f}s "
              f"({int(y_tr.sum()):,} train positives)")

    scores = {}
    for key in heads["T"]:
        scores[key] = 1.0 - (1.0 - heads["T"][key]) * (1.0 - heads["D"][key])
    return scores


def build_gbm_emissions(splits, eicu, emb_dir, device="cuda", seed=SEED):
    """Cache the GBM pseudo-agent's emissions for MIMIC splits and eICU.

    Not an :class:`Agent` subclass -- the fusion consumes it as a one-wide
    latent (the logit) plus a calibrated risk, exactly as
    ``scripts/orch_gbm_agent.py`` builds it. Reproduced here rather than
    imported because that script reads MIMIC-only cached scores
    (``baselines_fair.SCORE_CACHE``) and there is no such cache for eICU.

    Protocol matches the shipped agent: out-of-fold scores on train (so the
    fusion never sees in-sample GBM output), a booster fit on all of train for
    val/test/eICU, and isotonic calibration on val. The eICU design matrix is
    built from the eICU lab panel but scored by the MIMIC-fit booster and
    mapped through the MIMIC-fit calibrator -- nothing is refit.
    """
    from agents.base import AgentOutput
    from baselines_fair import build_design, variant_columns
    from distill_lead import fit_gbm_student, gbm_predict
    from modality_features import load_labs
    from sklearn.isotonic import IsotonicRegression

    cols = variant_columns()["FAIR+TRAJ"]
    labs_m, labs_e = load_labs(), load_labs(EICU_LABS)
    ordered = {k: v.sort_values(["stay_id", "hour"]).reset_index(drop=True)
               for k, v in splits.items()}
    y = {k: v[LABEL].to_numpy(float) for k, v in ordered.items()}

    print("[gbm] building design matrices...")
    X = {k: build_design(splits[k], labs_m, None)[0][cols] for k in splits}
    t0 = time.time()
    booster = fit_gbm_student(X["train"], y["train"], np.ones(len(y["train"])),
                              X["val"], y["val"], device=device, seed=seed)
    print(f"[gbm] booster fit in {time.time()-t0:.0f}s")

    # Out-of-fold train scores, split by PATIENT so no patient is scored by a
    # booster that saw them (the same two-fold scheme as the shipped agent).
    pid = ordered["train"]["patient_id"].to_numpy()
    patients = np.sort(ordered["train"]["patient_id"].unique())
    perm = np.random.default_rng(seed).permutation(patients)
    half = len(patients) // 2
    oof = np.full(len(pid), np.nan)
    for hold in (set(perm[:half]), set(perm[half:])):
        m = np.isin(pid, list(hold))
        b = fit_gbm_student(X["train"][~m], y["train"][~m],
                            np.ones(int((~m).sum())), X["val"], y["val"],
                            device=device, seed=seed)
        oof[m] = gbm_predict(b, X["train"][m])
    assert not np.isnan(oof).any()

    raw = {"train": oof,
           "val": gbm_predict(booster, X["val"]),
           "test": gbm_predict(booster, X["test"])}
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw["val"], y["val"])

    del X
    wanted = ["train", "val", "test"]
    eicu_ordered = None
    if eicu is not None:
        eicu_ordered = eicu.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        X_e = build_design(eicu, labs_e, None)[0][cols]
        raw["eicu"] = gbm_predict(booster, X_e)
        del X_e
        wanted.append("eicu")

    for split in wanted:
        keys = (eicu_ordered if split == "eicu" else ordered[split])[["stay_id", "hour"]]
        r = np.clip(raw[split], 1e-6, 1 - 1e-6)
        AgentOutput(keys=keys.copy(),
                    latent=np.log(r / (1 - r)).astype(np.float32)[:, None],
                    risk=iso.transform(raw[split]).astype(np.float32)
                    ).save(emb_dir / f"gbm_{split}.npz")
    print(f"[gbm] cached {', '.join('gbm_' + w for w in wanted)}.npz")


def build_agent(name, epochs, seed=SEED):
    """Construct one untrained agent by name.

    ``seed`` reaches every learned agent. NEWS2 takes none -- it is a
    deterministic lookup, so re-seeding it would be meaningless rather than
    merely useless.
    """
    if name == "vitals_tcn":
        from agents.vitals_tcn import VitalsTCNAgent
        return VitalsTCNAgent(max_epochs=epochs, seed=seed)
    if name == "news2":
        from agents.news2_agent import News2Agent
        return News2Agent()
    if name == "labs":
        from agents.labs_agent import LabsAgent
        return LabsAgent(max_epochs=epochs, seed=seed)
    if name == "ecg_meas":
        from agents.ecg_meas_agent import EcgMeasAgent
        return EcgMeasAgent(max_epochs=epochs, seed=seed)
    if name == "fluid":
        from agents.tabular_agent import FluidAgent
        return FluidAgent(max_epochs=epochs, seed=seed)
    if name == "resp":
        from agents.tabular_agent import RespAgent
        return RespAgent(max_epochs=epochs, seed=seed)
    if name == "cadence":
        # MIMIC-ONLY. eICU has no faithful cadence mapping: no order-entry
        # table (so n_orders_lab / n_orders_imaging would be structurally
        # zero) and a microbiology table of only ~17k rows (so n_micro is
        # degenerate) -- 3 of 9 channels absent by construction rather than by
        # hospital variation. Zero-filling them would depress the transfer
        # number for a reason that has nothing to do with generalisation.
        # Usable with --mimic-only, to price cadence's contribution on home
        # turf; point_at_eicu deliberately has no branch for it.
        from agents.tabular_agent import CadenceAgent
        return CadenceAgent(max_epochs=epochs, seed=seed)
    raise KeyError(f"agent {name!r} has no eICU feature mapping yet")


def point_at_eicu(name, agent):
    """Repoint an agent's auxiliary feature source at its eICU twin.

    Only the DATA source moves. Weights, normalisation statistics and the
    isotonic calibrator all stay as fitted on MIMIC -- repointing those would
    turn a transfer test into a refit.
    """
    from modality_features import load_fluids, load_labs, load_resp
    if name == "labs":
        agent.labs = load_labs(EICU_LABS)
    elif name == "fluid":
        agent.source = load_fluids(EICU_FLUIDS)
    elif name == "resp":
        agent.source = load_resp(EICU_RESP)


def restore_mimic(name, agent):
    from modality_features import load_fluids, load_labs, load_resp
    if name == "labs":
        agent.labs = load_labs()
    elif name == "fluid":
        agent.source = load_fluids()
    elif name == "resp":
        agent.source = load_resp()


def row(tag, y, r, extra=""):
    p = float(y.mean())
    a = auprc(y, r)
    print(f"  {tag:<30} n={len(y):>9,}  prev {100*p:5.2f}%  "
          f"AUPRC {a:.4f}  lift {a/p:5.2f}  AUROC {auroc(y, r):.4f}  "
          f"ECE {ece(y, r):.4f}{extra}")
    return dict(tag=tag, n=len(y), prev=p, auprc=a, lift=a / p,
                auroc=auroc(y, r), ece=ece(y, r))


def fresh_mask(df, bound):
    m = np.ones(len(df), dtype=bool)
    for c in VITALS:
        m &= df[f"{c}_hrs_since"].to_numpy() <= bound
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["vitals_tcn", "news2", "labs"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fusion-epochs", type=int, default=8)
    ap.add_argument("--no-check-repro", action="store_true")
    ap.add_argument("--reuse-emissions", action="store_true",
                    help="skip agent fitting when every needed emission .npz "
                         "is already cached under embeddings_eicu/. The agents "
                         "are deterministic given the MIMIC fit, so this only "
                         "skips recomputation -- it never reuses a stale eICU "
                         "source, because repointing happens at emit time.")
    ap.add_argument("--two-head", action="store_true",
                    help="also score the SHIPPED FLAGSHIP: treatment head + "
                         "mortality head combined by noisy-OR (T9). This is "
                         "the configuration the claims rest on.")
    ap.add_argument("--seed", type=int, default=42,
                    help="fusion seed. Follows T9/T18: seeds vary the FUSION, "
                         "not the agents -- the bank is fitted once at 42 and "
                         "shared, so a seed sweep costs the head fits only.")
    ap.add_argument("--reseed-agents", action="store_true",
                    help="re-fit the AGENTS at --seed too, not just the fusion. "
                         "The default follows T9/T18: the bank is fitted once "
                         "at 42 and shared, so a seed sweep measures fusion "
                         "variance only. This measures the whole stack's, at "
                         "roughly 10x the cost, and writes to its own "
                         "embeddings_eicu_s<seed>/ so runs cannot clobber "
                         "each other.")
    ap.add_argument("--label", default="run",
                    help="suffix for the cached score files")
    ap.add_argument("--mimic-only", action="store_true",
                    help="fit and report on MIMIC alone, emitting nothing on "
                         "eICU. Used to price a modality that eICU cannot "
                         "supply (cadence) on home turf, so its absence from "
                         "the transfer config is a known cost rather than a "
                         "confound.")
    args = ap.parse_args()

    global EMB_DIR
    if args.reseed_agents:
        EMB_DIR = REPO / "data" / "cache" / f"embeddings_eicu_s{args.seed}"
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[emb] {EMB_DIR.name}/"
          + (f"  (agents re-seeded at {args.seed})" if args.reseed_agents
             else "  (bank shared from seed 42; fusion seed only)"))
    splits = load_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    eicu = pd.read_parquet(EICU_COHORT)

    # A collision would make eICU stays inherit MIMIC ECG/labs rows silently.
    mimic_ids = set(pd.concat([train, val, test])["stay_id"].unique())
    clash = mimic_ids & set(eicu["stay_id"].unique())
    assert not clash, f"{len(clash)} eICU/MIMIC stay_id collisions"
    print(f"[data] MIMIC test {len(test):,} h | eICU {len(eicu):,} h "
          f"({eicu.stay_id.nunique():,} stays, {eicu.hospitalid.nunique()} hospitals)")

    # ---- agents: fit on MIMIC, emit on MIMIC and eICU --------------------
    need = ["train", "val", "test"] + ([] if args.mimic_only else ["eicu"])
    cached = all((EMB_DIR / f"{a}_{sp}.npz").exists()
                 for a in args.agents for sp in need)
    if args.reuse_emissions and cached:
        print(f"[reuse] every emission cached under {EMB_DIR.name}/; "
              f"skipping agent fits")
        args.agents = list(args.agents)
    elif "gbm" in args.agents:
        build_gbm_emissions({"train": train, "val": val, "test": test},
                            None if args.mimic_only else eicu, EMB_DIR,
                            seed=args.seed if args.reseed_agents else SEED)
    for name in ([] if (args.reuse_emissions and cached)
                 else [a for a in args.agents if a != "gbm"]):
        t0 = time.time()
        agent = build_agent(name, args.epochs,
                            seed=args.seed if args.reseed_agents else SEED)
        agent.fit(train, val)
        for split, df in (("train", train), ("val", val), ("test", test)):
            agent.emit_and_cache(df, EMB_DIR / f"{name}_{split}.npz")
        check_repro = not args.no_check_repro and not (
            args.reseed_agents and args.seed != SEED)
        if check_repro and (SHIPPED_EMB / f"{name}_test.npz").exists():
            from agents.base import AgentOutput
            mine = AgentOutput.load(EMB_DIR / f"{name}_test.npz")
            ship = AgentOutput.load(SHIPPED_EMB / f"{name}_test.npz")
            if mine.keys.equals(ship.keys):
                r = float(np.corrcoef(mine.risk, ship.risk)[0, 1])
                d = float(np.abs(mine.risk - ship.risk).max())
                print(f"[repro] {name}: risk corr vs shipped cache {r:.4f}, max|delta| {d:.4f}")
            else:
                print(f"[repro] {name}: key sets differ from shipped cache; skipped")
        if not args.mimic_only:
            point_at_eicu(name, agent)
            agent.emit_and_cache(eicu, EMB_DIR / f"{name}_eicu.npz")
            restore_mimic(name, agent)
        print(f"[agent] {name} done in {time.time()-t0:.0f}s")

    # ---- fusion: fit on MIMIC, score MIMIC test and eICU -----------------
    from orchestrator.fusion import AgentBank, Orchestrator
    bank_splits = ("train", "val", "test")
    if not args.mimic_only:
        bank_splits = bank_splits + ("eicu",)
    bank = AgentBank.load(splits=bank_splits,
                          agent_names=tuple(args.agents), emb_dir=EMB_DIR)
    orch = Orchestrator(bank=bank, max_epochs=args.fusion_epochs, seed=args.seed)
    t0 = time.time()
    orch.fit(train, val)
    print(f"[fusion] {'+'.join(args.agents)} fit in {time.time()-t0:.0f}s "
          f"(seed {args.seed})")

    def scored(df):
        out = orch.emit(df)
        srt = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
        assert (out.keys["stay_id"].to_numpy() == srt["stay_id"].to_numpy()).all()
        return srt, out.risk

    ts, tr = scored(test)
    if args.mimic_only:
        print(f"\n=== fusion ({'+'.join(args.agents)}) — MIMIC only ===")
        row("MIMIC test", ts[LABEL].to_numpy(float), tr)
        for b in (3, 2):
            mm = fresh_mask(ts, b)
            row(f"MIMIC  fresh <= {b}h", ts.loc[mm, LABEL].to_numpy(float), tr[mm])
        return
    es, er = scored(eicu)
    tag = "+".join(args.agents)
    np.savez_compressed(REPO / "data" / "cache" / f"eicu_fusion_{args.label}.npz",
                        stay_id=es.stay_id.to_numpy(), hour=es.hour.to_numpy(),
                        risk=er, agents=tag)
    report_transfer(f"single-head fusion ({tag})", ts, tr, es, er)

    if args.two_head:
        sc = two_head_scores(bank, {"train": train, "val": val, "test": test},
                             eicu, args.fusion_epochs, seed=args.seed)
        np.savez_compressed(
            REPO / "data" / "cache" / f"eicu_twohead_{args.label}.npz",
            stay_id=es.stay_id.to_numpy(), hour=es.hour.to_numpy(),
            risk=sc["eicu"], agents=tag,
            mimic_stay_id=ts.stay_id.to_numpy(), mimic_hour=ts.hour.to_numpy(),
            mimic_risk=sc["test"])
        report_transfer(f"TWO-HEAD flagship ({tag})", ts, sc["test"], es, sc["eicu"])


if __name__ == "__main__":
    main()
