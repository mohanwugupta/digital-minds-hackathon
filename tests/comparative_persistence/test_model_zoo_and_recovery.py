from analysis.comparative_persistence.hazard_models.baselines import MODEL_SPECS
from analysis.comparative_persistence.synthetic.generators import (
    generate_history_data,
    generate_latent_data,
    generate_sharing_data,
)
from analysis.comparative_persistence.synthetic.recovery import (
    recover_history_lag,
    recover_latent_rho,
    recover_sharing,
)


def test_required_interpretable_and_flexible_models_are_registered():
    required = {
        "intercept",
        "time_only",
        "immediate_state",
        "finite_history",
        "exponential_reward",
        "perseveration",
        "dual_history",
        "dynamic_reevaluation",
        "dynamic_reevaluation_oracle",
        "option_termination",
        "competitive_time_reward",
        "latent_commitment",
        "sunk_extension",
        "flexible_linear",
        "mlp",
        "gru",
    }
    assert required <= set(MODEL_SPECS)


def test_shared_and_task_specific_generators_recover_their_sharing_level():
    shared = generate_sharing_data("shared", episodes_per_task=100, seed=3)
    specific = generate_sharing_data("task_specific", episodes_per_task=100, seed=4)
    assert recover_sharing(shared) == "fully_shared"
    assert recover_sharing(specific) == "task_specific"


def test_known_five_lag_kernel_recovers_five_step_history():
    records = generate_history_data(lag=5, episodes=220, seed=5)
    assert recover_history_lag(records, candidates=(1, 2, 3, 5, 8)) == 5


def test_latent_recovery_and_zero_rho_collapse():
    persistent = generate_latent_data(rho=0.9, episodes=220, seed=6)
    immediate = generate_latent_data(rho=0.0, episodes=220, seed=7)
    assert recover_latent_rho(persistent, candidates=(0.0, 0.5, 0.9)) == 0.9
    assert recover_latent_rho(immediate, candidates=(0.0, 0.5, 0.9)) == 0.0
