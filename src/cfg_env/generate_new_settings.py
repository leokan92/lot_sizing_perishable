"""Generate setting_16.json ... setting_49.json from base templates.

Reproduces the 34 new experimental scenarios (Small / Moderate / Big tiers)
defined in the project plan. Each variant is a single delta applied to a
per-tier base configuration; "Combined" variants stack the H-Demand and
H-Supply deltas explicitly.

Run from the repo root:
    python src/cfg_env/generate_new_settings.py
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Callable

import numpy as np


CFG_DIR = os.path.dirname(os.path.abspath(__file__))

SMALL_BASE_FILE = os.path.join(CFG_DIR, "setting_1.json")
MODERATE_BASE_FILE = os.path.join(CFG_DIR, "setting_10.json")
BIG_BASE_FILE = os.path.join(CFG_DIR, "setting_15.json")

PATTERN_LENGTH_SMALL_MODERATE = 10


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_base(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def write_cfg(cfg: dict, idx: int) -> str:
    out_path = os.path.join(CFG_DIR, f"setting_{idx}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=4)
    return out_path


# ---------------------------------------------------------------------------
# Cardinality / structural deltas
# ---------------------------------------------------------------------------

def trim_items_suppliers(cfg: dict, n_items: int, n_suppliers: int) -> dict:
    """Trim a base config down to a smaller (n_items, n_suppliers) shape.

    Takes the first n_items rows of every per-item matrix and the first
    n_suppliers columns of every per-supplier matrix. Asserts each item still
    has at least one supplier after the trim.
    """
    cfg = copy.deepcopy(cfg)
    cfg["n_items"] = n_items
    cfg["n_suppliers"] = n_suppliers

    item_supplier_matrix = np.array(cfg["item_supplier_matrix"])[:n_items, :n_suppliers]
    coverage = item_supplier_matrix.sum(axis=1)
    assert coverage.min() >= 1, (
        f"Trim leaves some items with no supplier: coverage={coverage.tolist()}"
    )
    cfg["item_supplier_matrix"] = item_supplier_matrix.tolist()

    for key in (
        "unit_purchase_costs",
        "lead_times",
        "prob_full_fulfillment",
        "partial_fulfillment_beta_alpha",
        "partial_fulfillment_beta_beta",
    ):
        arr = np.array(cfg[key])[:n_items, :n_suppliers]
        cfg[key] = arr.tolist()

    cfg["fixed_order_costs"] = list(cfg["fixed_order_costs"])[:n_suppliers]

    for key in (
        "max_inventory_level",
        "holding_costs",
        "lost_sales_costs",
    ):
        cfg[key] = list(cfg[key])[:n_items]

    cfg["initial_inventory_age"] = [row[:] for row in cfg["initial_inventory_age"][:n_items]]
    cfg["shelf_life_cdf"] = [row[:] for row in cfg["shelf_life_cdf"][:n_items]]

    mu = np.array(cfg["demand_distribution"]["mu"])[:n_items, :]
    sigma = np.array(cfg["demand_distribution"]["sigma"])[:n_items, :]
    cfg["demand_distribution"]["mu"] = mu.tolist()
    cfg["demand_distribution"]["sigma"] = sigma.tolist()

    return cfg


def apply_horizon(cfg: dict, T: int, pattern_length: int) -> dict:
    """Set time_horizon and reshape mu/sigma/seasonal_factor accordingly.

    Two conventions are supported via pattern_length:
      - pattern_length < T (e.g. 10 for Small/Moderate tier): mu/sigma kept at
        width pattern_length, seasonal_factor at length pattern_length. The
        env tiles the pattern.
      - pattern_length == T (Big tier convention): mu/sigma resized to width T,
        seasonal_factor tiled out to length T.
    """
    cfg = copy.deepcopy(cfg)
    cfg["time_horizon"] = T

    dd = cfg["demand_distribution"]
    cur_mu = np.array(dd["mu"])
    cur_sigma = np.array(dd["sigma"])
    cur_season = list(dd["seasonal_factor"]) if dd.get("seasonal_factor") else None

    n_items = cfg["n_items"]

    if pattern_length == cur_mu.shape[1]:
        new_mu = cur_mu
        new_sigma = cur_sigma
    else:
        # Take the first column of the current matrix (each item is constant
        # along the time axis in all base templates) and repeat it.
        new_mu = np.tile(cur_mu[:, :1], (1, pattern_length))
        new_sigma = np.tile(cur_sigma[:, :1], (1, pattern_length))

    if cur_season is not None and len(cur_season) != pattern_length:
        # Resample the existing pattern to the new pattern length by tiling /
        # truncating from a canonical length-10 cycle.
        canonical_10 = [1.0, 1.5, 2.0, 1.3, 1.0, 0.9, 0.4, 0.5, 1.0, 1.8]
        # Use canonical_10 if the current pattern matches it; otherwise tile
        # the current pattern.
        src = cur_season if len(cur_season) <= pattern_length else canonical_10
        tiled = (src * (pattern_length // len(src) + 1))[:pattern_length]
        cur_season = tiled

    dd["mu"] = new_mu.tolist()
    dd["sigma"] = new_sigma.tolist()
    if cur_season is not None:
        dd["seasonal_factor"] = cur_season

    return cfg


# ---------------------------------------------------------------------------
# Demand / supplier / cost deltas
# ---------------------------------------------------------------------------

def apply_high_demand_sigma(cfg: dict, factor: float = 4.0) -> dict:
    """Multiply sigma by `factor` (H-Demand)."""
    cfg = copy.deepcopy(cfg)
    sigma = np.array(cfg["demand_distribution"]["sigma"]) * factor
    cfg["demand_distribution"]["sigma"] = sigma.tolist()
    return cfg


def apply_supplier_reliability(cfg: dict, p_full: float) -> dict:
    """Set prob_full_fulfillment to p_full on nonzero entries.

    Also adjusts partial_fulfillment_beta_alpha for the lower-reliability
    profile to match what existing low-reliability configs use (alpha ~ 4-5
    instead of 9, looser partial-fulfillment beta).
    """
    cfg = copy.deepcopy(cfg)
    pmat = np.array(cfg["prob_full_fulfillment"])
    mask = pmat > 0.0
    pmat[mask] = p_full
    cfg["prob_full_fulfillment"] = pmat.tolist()

    if p_full < 0.92:
        alpha = np.array(cfg["partial_fulfillment_beta_alpha"])
        alpha = np.where((mask) & (alpha > 5.0), 4.0, alpha)
        cfg["partial_fulfillment_beta_alpha"] = alpha.tolist()
    return cfg


def apply_low_reliability_jittered(cfg: dict) -> dict:
    """Reproduce the H-Supply profile from setting_3 / setting_12 / setting_15.

    Existing low-reliability configs use a small jitter around 0.85 instead
    of a flat value. Apply that here so 'Combined' variants exactly match
    legacy semantics.
    """
    cfg = copy.deepcopy(cfg)
    pmat = np.array(cfg["prob_full_fulfillment"])
    mask = pmat > 0.0
    # Deterministic jitter in 0.83-0.89 based on (item, supplier) index
    jitter_grid = np.array([0.85, 0.86, 0.87, 0.88, 0.84, 0.83, 0.89])
    flat_idx = 0
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            if mask[i, j]:
                pmat[i, j] = float(jitter_grid[flat_idx % len(jitter_grid)])
                flat_idx += 1
    cfg["prob_full_fulfillment"] = pmat.tolist()

    alpha = np.array(cfg["partial_fulfillment_beta_alpha"])
    alpha = np.where((mask) & (alpha > 5.0), 4.0, alpha)
    cfg["partial_fulfillment_beta_alpha"] = alpha.tolist()
    return cfg


def scale_holding(cfg: dict, factor: float) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["holding_costs"] = (np.array(cfg["holding_costs"]) * factor).tolist()
    return cfg


def scale_lost_sales(cfg: dict, factor: float) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["lost_sales_costs"] = (np.array(cfg["lost_sales_costs"]) * factor).tolist()
    return cfg


def scale_fixed_order(cfg: dict, factor: float) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["fixed_order_costs"] = (np.array(cfg["fixed_order_costs"]) * factor).tolist()
    return cfg


def scale_mu(cfg: dict, factor: float) -> dict:
    cfg = copy.deepcopy(cfg)
    mu = np.array(cfg["demand_distribution"]["mu"]) * factor
    cfg["demand_distribution"]["mu"] = mu.tolist()
    return cfg


def scale_capacity(cfg: dict, factor: float) -> dict:
    cfg = copy.deepcopy(cfg)
    caps = (np.array(cfg["max_inventory_level"]) * factor).astype(int)
    cfg["max_inventory_level"] = caps.tolist()
    return cfg


# ---------------------------------------------------------------------------
# Lead-time deltas
# ---------------------------------------------------------------------------

def apply_lead_time_uniform(cfg: dict, lt: int) -> dict:
    """Set all positive lead-time entries to `lt`. Zero entries (no supplier
    for that item) remain zero, preserving the cost-mask invariant.
    """
    cfg = copy.deepcopy(cfg)
    mask = np.array(cfg["item_supplier_matrix"]) == 1
    lt_mat = np.zeros_like(mask, dtype=int)
    lt_mat[mask] = lt
    cfg["lead_times"] = lt_mat.tolist()
    return cfg


def apply_lead_time_heterogeneous(cfg: dict, values=(1, 4, 5, 6)) -> dict:
    """Sprinkle longer / heterogeneous lead times for LongLT variants.

    Walks the supplier mask in row-major order and assigns successive values
    from the given pool cyclically.
    """
    cfg = copy.deepcopy(cfg)
    mask = np.array(cfg["item_supplier_matrix"]) == 1
    lt_mat = np.zeros_like(mask, dtype=int)
    idx = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                lt_mat[i, j] = values[idx % len(values)]
                idx += 1
    cfg["lead_times"] = lt_mat.tolist()
    return cfg


# ---------------------------------------------------------------------------
# Shelf-life deltas
# ---------------------------------------------------------------------------

def _build_shelf_life_cdf_row(max_age: int) -> list:
    """A monotonic non-decreasing CDF row ending at 1.0 with mild leakage in
    earlier ages, consistent with the existing setting_0 base profile.
    """
    if max_age == 2:
        return [0.1, 1.0]
    if max_age == 3:
        return [0.05, 0.15, 1.0]
    if max_age == 4:
        return [0.05, 0.10, 0.15, 1.0]
    if max_age == 5:
        return [0.03, 0.05, 0.08, 0.12, 1.0]
    if max_age == 6:
        return [0.02, 0.04, 0.06, 0.09, 0.13, 1.0]
    raise ValueError(f"Unsupported max_age={max_age}")


def apply_max_age(cfg: dict, max_age: int) -> dict:
    """Change shelf life, rebuilding shelf_life_cdf and initial_inventory_age.

    Initial stock per item:
      - max_age 2 -> 6 (reduce to avoid massive synthetic first-step wastage)
      - max_age 3 -> 9
      - otherwise -> 12 (existing convention)
    """
    cfg = copy.deepcopy(cfg)
    n_items = cfg["n_items"]
    cfg["max_age"] = max_age

    cdf_row = _build_shelf_life_cdf_row(max_age)
    cfg["shelf_life_cdf"] = [cdf_row[:] for _ in range(n_items)]

    if max_age == 2:
        init_units = 6
    elif max_age == 3:
        init_units = 9
    else:
        init_units = 12
    cfg["initial_inventory_age"] = [
        [init_units] + [0] * (max_age - 1) for _ in range(n_items)
    ]
    return cfg


# ---------------------------------------------------------------------------
# Seasonality deltas
# ---------------------------------------------------------------------------

# Length-10 canonical patterns ("L" = strong cyclic from setting_1)
SEASON_FLAT = [1.0] * 10
SEASON_L_STRONG = [1.0, 1.5, 2.0, 1.3, 1.0, 0.9, 0.4, 0.5, 1.0, 1.8]
SEASON_MILD = [0.88, 0.95, 1.07, 1.10, 1.05, 1.16, 0.85, 1.00, 0.95, 0.99]
SEASON_BIMODAL = [0.6, 0.8, 1.4, 1.6, 1.0, 0.5, 0.8, 1.5, 1.3, 0.5]
SEASON_TREND = [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90]
SEASON_STRONG = [0.2, 0.6, 1.2, 1.8, 2.2, 2.5, 1.9, 1.3, 0.7, 0.3]


def apply_seasonal_pattern(cfg: dict, pattern: list) -> dict:
    """Replace the seasonal_factor (assumes mu/sigma already at the matching
    width — caller is responsible for invoking apply_horizon if changing T).
    """
    cfg = copy.deepcopy(cfg)
    pattern_length = len(pattern)

    dd = cfg["demand_distribution"]
    mu = np.array(dd["mu"])
    sigma = np.array(dd["sigma"])

    if mu.shape[1] != pattern_length:
        mu = np.tile(mu[:, :1], (1, pattern_length))
        sigma = np.tile(sigma[:, :1], (1, pattern_length))

    dd["mu"] = mu.tolist()
    dd["sigma"] = sigma.tolist()
    dd["seasonal_factor"] = list(pattern)
    return cfg


# ---------------------------------------------------------------------------
# Invariants — fail fast in the generator
# ---------------------------------------------------------------------------

def assert_invariants(cfg: dict, label: str) -> None:
    n_items = cfg["n_items"]
    n_suppliers = cfg["n_suppliers"]
    max_age = cfg["max_age"]

    mask = np.array(cfg["item_supplier_matrix"])
    assert mask.shape == (n_items, n_suppliers), f"[{label}] mask shape {mask.shape}"
    assert mask.sum(axis=1).min() >= 1, f"[{label}] item with no supplier"

    cost_mask = np.array(cfg["unit_purchase_costs"]) > 0
    assert np.array_equal(cost_mask, mask == 1), (
        f"[{label}] cost-mask mismatch with item_supplier_matrix"
    )

    lt_mask = np.array(cfg["lead_times"]) > 0
    assert np.array_equal(lt_mask, mask == 1), (
        f"[{label}] lead-time mask mismatch with item_supplier_matrix"
    )

    p_mask = np.array(cfg["prob_full_fulfillment"]) > 0
    assert np.array_equal(p_mask, mask == 1), (
        f"[{label}] prob_full mask mismatch with item_supplier_matrix"
    )

    assert len(cfg["fixed_order_costs"]) == n_suppliers
    assert len(cfg["holding_costs"]) == n_items
    assert len(cfg["lost_sales_costs"]) == n_items
    assert len(cfg["max_inventory_level"]) == n_items

    cdf = np.array(cfg["shelf_life_cdf"])
    assert cdf.shape == (n_items, max_age), f"[{label}] cdf shape {cdf.shape}"
    assert np.allclose(cdf[:, -1], 1.0), f"[{label}] cdf last col != 1.0"
    assert np.all(np.diff(cdf, axis=1) >= -1e-9), f"[{label}] cdf not monotonic"

    init_age = np.array(cfg["initial_inventory_age"])
    assert init_age.shape == (n_items, max_age), f"[{label}] init_age shape {init_age.shape}"

    dd = cfg["demand_distribution"]
    mu = np.array(dd["mu"])
    sigma = np.array(dd["sigma"])
    if dd.get("seasonal_factor"):
        pat_len = len(dd["seasonal_factor"])
    else:
        pat_len = mu.shape[1]
    assert mu.shape == (n_items, pat_len), f"[{label}] mu shape {mu.shape}, want ({n_items},{pat_len})"
    assert sigma.shape == (n_items, pat_len), f"[{label}] sigma shape {sigma.shape}"

    # Peak demand vs capacity — warn-level (not all variants need to satisfy,
    # but flag if peak > capacity which would force constant lost sales).
    if dd.get("seasonal_factor"):
        peak_season = max(dd["seasonal_factor"])
    else:
        peak_season = 1.0
    peak_mu = mu.max(axis=1) * peak_season
    caps = np.array(cfg["max_inventory_level"])
    if (peak_mu > caps).any():
        offenders = np.where(peak_mu > caps)[0].tolist()
        # Soft assert: scale-up should already have been applied by caller for
        # Combined / HighDem variants. If we hit this it's a real bug.
        raise AssertionError(
            f"[{label}] peak demand exceeds capacity for items {offenders}: "
            f"peak={peak_mu[offenders].tolist()} caps={caps[offenders].tolist()}"
        )


# ---------------------------------------------------------------------------
# Manifest — list of (idx, builder) entries
# ---------------------------------------------------------------------------

def build_all() -> dict:
    small_base = load_base(SMALL_BASE_FILE)
    moderate_base = load_base(MODERATE_BASE_FILE)
    big_base = load_base(BIG_BASE_FILE)

    out: dict[int, dict] = {}

    # -- Group 1: Small extensions (setting_16..35), base = setting_1 --
    out[16] = apply_max_age(small_base, 2)
    out[17] = apply_max_age(small_base, 3)
    out[18] = apply_max_age(small_base, 5)
    out[19] = apply_max_age(small_base, 6)

    out[20] = scale_holding(small_base, 10.0)
    out[21] = scale_lost_sales(small_base, 3.0)
    out[22] = scale_fixed_order(small_base, 3.0)

    out[23] = scale_mu(small_base, 0.5)
    high_dem = scale_mu(small_base, 2.0)
    high_dem = scale_capacity(high_dem, 2.0)
    out[24] = high_dem

    out[25] = apply_high_demand_sigma(small_base, factor=2.0)
    out[26] = apply_high_demand_sigma(small_base, factor=3.0)

    out[27] = apply_supplier_reliability(small_base, p_full=0.90)

    out[28] = apply_lead_time_uniform(small_base, lt=1)
    out[29] = apply_lead_time_uniform(small_base, lt=5)
    lt7 = apply_lead_time_uniform(small_base, lt=7)
    lt7 = scale_capacity(lt7, 1.5)  # absorb the longer LT buffer
    out[30] = lt7

    out[31] = apply_seasonal_pattern(small_base, SEASON_BIMODAL)
    out[32] = apply_seasonal_pattern(small_base, SEASON_TREND)
    # StrongSeas peaks at 2.5 — verify capacity headroom (small base mu max=6 ->
    # peak = 15, max_inventory_level = 30, ok).
    out[33] = apply_seasonal_pattern(small_base, SEASON_STRONG)

    # I-4-2-60-L: just extend horizon (keeps length-10 pattern, tiled)
    out[34] = apply_horizon(small_base, T=60, pattern_length=10)

    # I-4-2-90-H-Combined: T=90, sigma x 4, low reliability profile
    combined_small = apply_horizon(small_base, T=90, pattern_length=10)
    combined_small = apply_high_demand_sigma(combined_small, factor=4.0)
    combined_small = apply_low_reliability_jittered(combined_small)
    out[35] = combined_small

    # -- Group 2: Moderate (setting_36..45) --
    # Bases trimmed from setting_10 (10 items, 4 sup, T=30, max_age=4)
    base_6_3 = trim_items_suppliers(moderate_base, n_items=6, n_suppliers=3)
    base_8_3 = trim_items_suppliers(moderate_base, n_items=8, n_suppliers=3)

    out[36] = base_6_3

    combined_6_3 = apply_high_demand_sigma(base_6_3, factor=4.0)
    combined_6_3 = apply_low_reliability_jittered(combined_6_3)
    out[37] = combined_6_3

    base_6_3_T45 = apply_horizon(base_6_3, T=45, pattern_length=10)
    out[38] = apply_seasonal_pattern(base_6_3_T45, SEASON_MILD)

    out[39] = base_8_3

    out[40] = apply_high_demand_sigma(base_8_3, factor=4.0)

    base_8_3_T45 = apply_horizon(base_8_3, T=45, pattern_length=10)
    out[41] = apply_lead_time_heterogeneous(base_8_3_T45)

    out[42] = apply_lead_time_heterogeneous(moderate_base)
    out[43] = apply_seasonal_pattern(moderate_base, SEASON_MILD)
    out[44] = apply_max_age(moderate_base, 5)
    out[45] = apply_horizon(moderate_base, T=60, pattern_length=10)

    # -- Group 3: Big (setting_46..49) --
    # Big base = setting_15 (20 items, 5 sup, T=60). Big-tier convention:
    # mu/sigma/seasonal_factor at width = time_horizon.
    base_15_4 = trim_items_suppliers(big_base, n_items=15, n_suppliers=4)
    # Big-tier convention: mu/sigma/seasonal_factor at full T width.
    # Setting_15 already uses width-60 -> trimming preserves that.

    # I-15-4-30-L: trim + shorten horizon (full-T convention)
    out[46] = apply_horizon(base_15_4, T=30, pattern_length=30)

    # I-15-4-60-H-Combined: trim + keep T=60 + H-Demand + H-Supply
    big_47 = base_15_4  # already T=60 width 60
    big_47 = apply_high_demand_sigma(big_47, factor=4.0)
    big_47 = apply_low_reliability_jittered(big_47)
    out[47] = big_47

    # I-20-5-30-L: full big cardinality, shorter horizon, low-cv high-p_full
    big_48 = copy.deepcopy(big_base)
    big_48 = apply_horizon(big_48, T=30, pattern_length=30)
    # Lower sigma to "low cv" and raise p_full to >=0.95 (setting_15 base is
    # high sigma + low p_full).
    sigma_low = (np.array(big_48["demand_distribution"]["sigma"]) * 0.25).tolist()
    big_48["demand_distribution"]["sigma"] = sigma_low
    big_48 = apply_supplier_reliability(big_48, p_full=0.97)
    out[48] = big_48

    # I-20-5-90-H-Combined: full big cardinality, long horizon, H + H
    big_49 = copy.deepcopy(big_base)
    big_49 = apply_horizon(big_49, T=90, pattern_length=90)
    # Base sigma is already high-cv; ensure it stays high after horizon resize
    # (apply_horizon repeats column 0 of the original sigma which is already
    # high). Apply low-reliability profile for the "Combined" half.
    big_49 = apply_low_reliability_jittered(big_49)
    # Long horizon + high demand needs more capacity headroom.
    big_49 = scale_capacity(big_49, 1.3)
    out[49] = big_49

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INSTANCE_NAMES = {
    16: "I-4-2-30-L-SL2",
    17: "I-4-2-30-L-SL3",
    18: "I-4-2-30-L-SL5",
    19: "I-4-2-30-L-SL6",
    20: "I-4-2-30-L-HighHold",
    21: "I-4-2-30-L-HighShort",
    22: "I-4-2-30-L-HighFixed",
    23: "I-4-2-30-L-LowDem",
    24: "I-4-2-30-L-HighDem",
    25: "I-4-2-30-M-Demand",
    26: "I-4-2-30-MH-Demand",
    27: "I-4-2-30-M-Supply",
    28: "I-4-2-30-L-LT1",
    29: "I-4-2-30-L-LT5",
    30: "I-4-2-30-L-LT7",
    31: "I-4-2-30-L-Bimodal",
    32: "I-4-2-30-L-Trend",
    33: "I-4-2-30-L-StrongSeas",
    34: "I-4-2-60-L",
    35: "I-4-2-90-H-Combined",
    36: "I-6-3-30-L",
    37: "I-6-3-30-H-Combined",
    38: "I-6-3-45-L-MildSeas",
    39: "I-8-3-30-L",
    40: "I-8-3-30-H-Demand",
    41: "I-8-3-45-L-LongLT",
    42: "I-10-4-30-L-LongLT",
    43: "I-10-4-30-L-MildSeas",
    44: "I-10-4-30-L-SL5",
    45: "I-10-4-60-L",
    46: "I-15-4-30-L",
    47: "I-15-4-60-H-Combined",
    48: "I-20-5-30-L",
    49: "I-20-5-90-H-Combined",
}


def main() -> None:
    configs = build_all()
    for idx in sorted(configs.keys()):
        cfg = configs[idx]
        label = f"setting_{idx} ({INSTANCE_NAMES[idx]})"
        assert_invariants(cfg, label)
        path = write_cfg(cfg, idx)
        print(f"  wrote {path}  ({INSTANCE_NAMES[idx]})")
    print(f"\nDone: {len(configs)} configs generated.")


if __name__ == "__main__":
    main()
