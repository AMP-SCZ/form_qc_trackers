"""Randomized graph-contract tests for the calculated-field transformer.

These tests deliberately use a fixed seed so failures are reproducible while
covering much denser dependency shapes than the small example-based tests.
"""

from __future__ import annotations

import random

import pandas as pd

from process_variables import transform_calculated_fields as calculated


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        calculated.VARIABLE_COLUMN: [row[0] for row in rows],
        calculated.FORM_COLUMN: ["audit_form"] * len(rows),
        calculated.FIELD_TYPE_COLUMN: [row[1] for row in rows],
        calculated.CALCULATION_COLUMN: [row[2] for row in rows],
        calculated.IDENTIFIER_COLUMN: [
            "y" if row[0].startswith("identifier_") else "" for row in rows
        ],
    })


def test_seeded_random_dags_order_every_dependency_and_propagate_identifiers(
        ) -> None:
    randomizer = random.Random(0xCA1C)
    for trial in range(12):
        identifiers = [f"identifier_{index}" for index in range(5)]
        rows: list[tuple[str, str, str]] = [
            (identifier, "text", "") for identifier in identifiers
        ]
        direct_identifiers: dict[str, set[str]] = {}
        dependencies: dict[str, set[str]] = {}
        variables = [f"calc_{trial}_{index}" for index in range(80)]
        for index, variable in enumerate(variables):
            possible_calcs = variables[:index]
            calc_dependencies = set(randomizer.sample(
                possible_calcs,
                k=randomizer.randint(0, min(4, len(possible_calcs))),
            ))
            identifier_dependencies = set(randomizer.sample(
                identifiers, k=randomizer.randint(0, 2)))
            dependencies[variable] = calc_dependencies
            direct_identifiers[variable] = identifier_dependencies
            references = sorted(calc_dependencies | identifier_dependencies)
            expression = " + ".join(f"[{item}]" for item in references) or "1"
            rows.append((variable, "calc", expression))

        # Input order must not carry any hidden topological assumption.
        randomizer.shuffle(rows)
        transformer = calculated.TransformCalculatedFields(_frame(rows))
        entries = transformer()
        position = {
            variable: index
            for index, variable in enumerate(transformer.evaluation_order)
        }
        assert set(transformer.evaluation_order) == set(variables)
        for variable in variables:
            assert entries[variable]["status"] == "converted"
            for dependency in dependencies[variable]:
                assert position[dependency] < position[variable]

            expected = set(direct_identifiers[variable])
            stack = list(dependencies[variable])
            visited: set[str] = set()
            while stack:
                dependency = stack.pop()
                if dependency in visited:
                    continue
                visited.add(dependency)
                expected.update(direct_identifiers[dependency])
                stack.extend(dependencies[dependency])
            actual = (
                set(entries[variable]["direct_identifier_dependencies"])
                | set(entries[variable]["transitive_identifier_dependencies"])
            )
            assert actual == expected


def test_seeded_random_cycles_only_label_cycle_members_and_block_descendants(
        ) -> None:
    randomizer = random.Random(0xC1C1E)
    for trial in range(20):
        cycle_size = randomizer.randint(1, 12)
        cycle = [f"cycle_{trial}_{index}" for index in range(cycle_size)]
        rows: list[tuple[str, str, str]] = []
        for index, variable in enumerate(cycle):
            dependency = cycle[(index + 1) % cycle_size]
            rows.append((variable, "calc", f"[{dependency}] + 1"))

        downstream = [f"downstream_{trial}_{index}" for index in range(20)]
        previous = cycle[0]
        for variable in downstream:
            rows.append((variable, "calc", f"[{previous}] + 1"))
            previous = variable
        independent = [f"independent_{trial}_{index}" for index in range(20)]
        rows.extend((variable, "calc", "1") for variable in independent)
        randomizer.shuffle(rows)

        transformer = calculated.TransformCalculatedFields(_frame(rows))
        entries = transformer()
        assert set(transformer.evaluation_order) == set(independent)
        assert all(
            entries[variable]["error_code"] == "calculation_dependency_cycle"
            for variable in cycle
        )
        assert all(
            entries[variable]["error_code"] == "blocked_calculation_dependency"
            for variable in downstream
        )
