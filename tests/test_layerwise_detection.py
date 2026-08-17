from analysis.layerwise_detection import summarize_layerwise_detection


def synthetic_layers():
    p_values = [0.20, 0.04, 0.001, 0.001]
    normalized = [0.05, 0.20, 0.60, 1.00]
    rows = []
    for layer, (p_value, effect) in enumerate(zip(p_values, normalized)):
        rows.append(
            {
                "layer": layer,
                "stop": {
                    "raw_slope": -effect,
                    "normal_approximation_p_value": p_value,
                },
                "continue": {
                    "raw_slope": effect,
                    "normal_approximation_p_value": p_value,
                },
                "relative_incentive_r_squared": 0.1 * (layer + 1),
                "normalized_to_behavior": {
                    "stop_raw_slope": effect,
                    "continue_raw_slope": effect,
                    "relative_incentive_r_squared": 0.25 * (layer + 1),
                },
            }
        )
    return rows


def test_detection_summary_reports_nominal_corrected_and_sustained_onsets():
    summary = summarize_layerwise_detection(synthetic_layers(), alpha=0.05)

    assert summary["stop"]["first_nominal_detectable_layer"] == 1
    assert summary["continue"]["first_nominal_detectable_layer"] == 1
    assert summary["stop"]["first_holm_detectable_layer"] == 2
    assert summary["continue"]["first_holm_detectable_layer"] == 2
    assert summary["stop"]["first_sustained_nominal_layer"] == 1
    assert summary["relative_incentive"]["first_half_final_layer"] == 1


def test_detection_summary_describes_growth_toward_final_layer():
    summary = summarize_layerwise_detection(synthetic_layers(), alpha=0.05)

    assert summary["stop"]["final_layer"] == 3
    assert summary["stop"]["final_behavior_normalized_effect"] == 1.0
    assert summary["stop"]["largest_absolute_growth_step"]["to_layer"] == 3
    assert summary["relative_incentive"]["final_r_squared"] == 0.4
