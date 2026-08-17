"""Pure helpers for declaring and summarizing layerwise effect detectability."""


def _holm_rejections(layer_p_values: list[tuple[int, float]], alpha: float) -> set[int]:
    ordered = sorted(layer_p_values, key=lambda item: (item[1], item[0]))
    rejected = set()
    total = len(ordered)
    for rank, (layer, p_value) in enumerate(ordered):
        if p_value <= alpha / (total - rank):
            rejected.add(layer)
        else:
            break
    return rejected


def _first_sustained(ordered_layers: list[int], detectable: set[int]):
    for index, layer in enumerate(ordered_layers):
        if all(candidate in detectable for candidate in ordered_layers[index:]):
            return layer
    return None


def _largest_growth(rows: list[dict], key: str) -> dict | None:
    best = None
    for previous, current in zip(rows, rows[1:]):
        left = previous["normalized_to_behavior"][key]
        right = current["normalized_to_behavior"][key]
        if left is None or right is None:
            continue
        growth = abs(float(right)) - abs(float(left))
        candidate = {
            "from_layer": int(previous["layer"]),
            "to_layer": int(current["layer"]),
            "absolute_effect_growth": growth,
        }
        if best is None or (growth, candidate["to_layer"]) >= (
            best["absolute_effect_growth"],
            best["to_layer"],
        ):
            best = candidate
    return best


def _effect_summary(
    rows: list[dict], effect: str, expected_sign: int, alpha: float
) -> dict:
    ordered_layers = [int(row["layer"]) for row in rows]
    layer_p_values = []
    nominal = set()
    for row in rows:
        layer = int(row["layer"])
        coefficient = row[effect]
        p_value = coefficient.get("normal_approximation_p_value")
        correct_sign = expected_sign * float(coefficient["raw_slope"]) > 0
        if p_value is not None:
            layer_p_values.append((layer, float(p_value)))
            if correct_sign and float(p_value) < alpha:
                nominal.add(layer)
    holm_all = _holm_rejections(layer_p_values, alpha)
    holm = {
        int(row["layer"])
        for row in rows
        if int(row["layer"]) in holm_all
        and expected_sign * float(row[effect]["raw_slope"]) > 0
    }
    normalized_key = f"{effect}_raw_slope"
    final = rows[-1]
    return {
        "expected_sign": "negative" if expected_sign < 0 else "positive",
        "alpha_two_sided": alpha,
        "multiple_comparison_correction": "Holm-Bonferroni within effect across layers",
        "nominal_detectable_layers": sorted(nominal),
        "holm_detectable_layers": sorted(holm),
        "first_nominal_detectable_layer": min(nominal) if nominal else None,
        "first_holm_detectable_layer": min(holm) if holm else None,
        "first_sustained_nominal_layer": _first_sustained(ordered_layers, nominal),
        "first_sustained_holm_layer": _first_sustained(ordered_layers, holm),
        "final_layer": int(final["layer"]),
        "final_raw_slope": float(final[effect]["raw_slope"]),
        "final_standardized_beta": float(
            final[effect].get("standardized_beta", final[effect]["raw_slope"])
        ),
        "final_behavior_normalized_effect": final["normalized_to_behavior"][
            normalized_key
        ],
        "largest_absolute_growth_step": _largest_growth(rows, normalized_key),
    }


def _first_at_fraction(rows: list[dict], final_value: float, fraction: float):
    threshold = fraction * final_value
    for row in rows:
        if float(row["relative_incentive_r_squared"]) >= threshold:
            return int(row["layer"])
    return None


def summarize_layerwise_detection(
    layers: list[dict], *, alpha: float = 0.05
) -> dict:
    """Summarize onset while keeping nominal and multiplicity-adjusted claims separate."""
    if not layers:
        raise ValueError("at least one layer is required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must fall in (0, 1)")
    rows = sorted(layers, key=lambda row: int(row["layer"]))
    if len({int(row["layer"]) for row in rows}) != len(rows):
        raise ValueError("layer rows must be unique")
    final_r_squared = float(rows[-1]["relative_incentive_r_squared"])
    return {
        "interpretation": (
            "Detectability marks where an incentive-induced difference is expressed "
            "along a frozen direction; it does not locate the computation."
        ),
        "stop": _effect_summary(rows, "stop", -1, alpha),
        "continue": _effect_summary(rows, "continue", 1, alpha),
        "relative_incentive": {
            "final_layer": int(rows[-1]["layer"]),
            "final_r_squared": final_r_squared,
            "final_behavior_normalized_r_squared": rows[-1][
                "normalized_to_behavior"
            ]["relative_incentive_r_squared"],
            "first_quarter_final_layer": _first_at_fraction(
                rows, final_r_squared, 0.25
            ),
            "first_half_final_layer": _first_at_fraction(
                rows, final_r_squared, 0.50
            ),
            "first_three_quarters_final_layer": _first_at_fraction(
                rows, final_r_squared, 0.75
            ),
            "largest_absolute_growth_step": _largest_growth(
                rows, "relative_incentive_r_squared"
            ),
        },
    }
