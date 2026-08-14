# Held-out probe mechanism diagnostic

Best layer: **2**; sparse neurons: **26**
Test data: **78 episodes / 1800 states**

| Analysis | States | Probe beta | Cluster SE | Delta R² | p (normal approx.) |
|---|---:|---:|---:|---:|---:|
| Recent-history controls | 1800 | 0.016 | 0.048 | 0.000 | 0.74 |
| Controls + cumulative score | 1800 | 0.213 | 0.040 | 0.016 | 1.14e-07 |
| Previous outcome fixed at loss | 809 | 0.013 | 0.046 | 0.000 | 0.777 |
| Previous outcome fixed at gain | 913 | 0.027 | 0.064 | 0.001 | 0.67 |

Primary controls are previous outcome, an initial-state indicator, loss streak, and nonlinear round terms. Standard errors use episode clusters. Layer and sparse-neuron selection used validation data; this diagnostic uses test episodes only.

## Evidence flags

- probe_adds_beyond_recent_history: **False**
- probe_adds_within_last_loss_states: **False**
- probe_encodes_history_within_last_loss_states: **True**
- integrated_value_pattern_supported: **False**

These held-out associations distinguish integrated value from a recent-outcome heuristic but are not causal; activation steering provides the causal test.
