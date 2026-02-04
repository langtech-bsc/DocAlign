"""Threshold presets for parallel vs comparable data modes.

Every value can be overridden individually in the YAML config under a
``thresholds:`` key.
"""

DEFAULTS = {
    "parallel": {
        "para_sim_threshold": 0.30,
        "sent_window": 20,
        "sent_baseline_threshold": 0.70,
        "sent_neighbour_margin": 0.08,
        "sent_fuzzy_threshold": 0.55,
        "sent_min_span_score": 0.60,
        "min_chunk_score": 0.0,
    },
    "comparable": {
        "para_sim_threshold": 0.45,
        "sent_window": 10,
        "sent_baseline_threshold": 0.75,
        "sent_neighbour_margin": 0.05,
        "sent_fuzzy_threshold": 0.65,
        "sent_min_span_score": 0.65,
        "min_chunk_score": 0.65,
    },
}


def resolve_thresholds(data_mode="parallel", overrides=None):
    """Return a flat dict of thresholds.

    Starts from mode defaults, then applies any per-key overrides from
    the config.
    """
    base = dict(DEFAULTS.get(data_mode, DEFAULTS["parallel"]))
    if overrides:
        for key in base:
            if key in overrides:
                base[key] = overrides[key]
    return base
