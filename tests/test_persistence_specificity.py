from analysis.persistence_specificity import classify_candidate


def test_label_only_candidate_is_not_persistence_specific():
    result = classify_candidate(
        persistence_sensitivity=0.2,
        cross_manipulation_transfer=0.1,
        cross_task_transfer=0.1,
        nuisance_sensitivity={
            "label": 0.9,
            "arbitrary_choice": 0.1,
            "terminality": 0.1,
            "generic_value": 0.1,
        },
        minimum_transfer=0.25,
        maximum_nuisance_fraction=0.5,
    )
    assert result["classification"] == "no_persistence_specific_candidate"
    assert result["criteria"]["label_specificity"] is False


def test_generic_decision_only_search_returns_explicit_null():
    result = classify_candidate(
        persistence_sensitivity=0.8,
        cross_manipulation_transfer=0.7,
        cross_task_transfer=0.7,
        nuisance_sensitivity={
            "label": 0.7,
            "arbitrary_choice": 0.8,
            "terminality": 0.75,
            "generic_value": 0.82,
        },
        minimum_transfer=0.25,
        maximum_nuisance_fraction=0.5,
    )
    assert result["classification"] == "no_persistence_specific_candidate"
    assert result["alternative_hypothesis"] == "domain_general_decision_or_value"

