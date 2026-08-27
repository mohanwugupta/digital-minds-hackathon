"""Observable/oracle recurrent commitment feature definitions."""

from __future__ import annotations


def commitment_inputs(information_set: str) -> tuple[str, ...]:
    if information_set == "observable":
        return ("relative_value", "cost_pressure", "progress_evidence")
    if information_set == "oracle":
        return ("oracle_relative_value", "cost_pressure", "progress_evidence")
    raise ValueError("information_set must be observable or oracle")


def generic_value_input(information_set: str) -> str:
    if information_set == "observable":
        return "relative_value"
    if information_set == "oracle":
        return "oracle_relative_value"
    raise ValueError("information_set must be observable or oracle")
