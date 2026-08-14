# Held-out probe mechanism diagnostic

Best layer: **17**; sparse neurons: **26**
Test data: **78 episodes / 1800 states**

| Analysis | States | Probe beta | Cluster SE | Delta R² | p (normal approx.) |
|---|---:|---:|---:|---:|---:|
| Recent-history controls | 1800 | -0.099 | 0.040 | 0.003 | 0.014 |
| Controls + cumulative score | 1800 | -0.066 | 0.043 | 0.001 | 0.123 |
| Previous outcome fixed at loss | 809 | -0.179 | 0.063 | 0.008 | 0.00444 |
| Previous outcome fixed at gain | 913 | -0.073 | 0.050 | 0.002 | 0.15 |

Primary controls are previous outcome, an initial-state indicator, loss streak, and nonlinear round terms. Standard errors use episode clusters. Layer and sparse-neuron selection used validation data; this diagnostic uses test episodes only.

## Evidence flags

- probe_adds_beyond_recent_history: **False**
- probe_adds_within_last_loss_states: **False**
- probe_encodes_history_within_last_loss_states: **False**
- integrated_value_pattern_supported: **False**

These held-out associations distinguish integrated value from a recent-outcome heuristic but are not causal; activation steering provides the causal test.
