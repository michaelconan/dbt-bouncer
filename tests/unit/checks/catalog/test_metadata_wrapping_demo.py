"""Demonstration of the "wrapped catalog metadata" bug and its fix.

This module is a standalone, narrated companion to
``test_catalog_sources.py``. It does not add new coverage; instead it
reproduces the bug in isolation so the failure mode is obvious without
needing to understand the full check-framework machinery.

Background
----------
``check_source_columns_are_all_documented`` compares two things for the same
source table:

* The *actual* columns, from ``catalog.json`` (via the ``catalog_source``
  parameter, which the check-framework auto-unwraps if the object exposes an
  attribute literally named after the parameter - it does not here, so this
  side is always the flat ``CatalogNodeEntry`` dbt-bouncer's own parser
  builds from ``catalog.json``, with columns directly on it).
* The *documented* columns, from ``manifest.json`` (via ``ctx.sources``).

dbt-bouncer's own parser wraps every manifest source in a container object
that nests the real, flat source one level down under a ``.source``
attribute - the container itself has no ``columns`` (or ``unique_id``, in the
original bug) of its own. A naive lookup like
``next(s for s in ctx.sources if s.unique_id == ...)`` returns the *wrapper*,
not the real source, so the subsequent ``source.columns`` raises
``AttributeError: 'SimpleNamespace' object has no attribute 'columns'``
instead of reporting a pass/fail governance finding.

The project's own unit-test helper (``dbt_bouncer.testing.check_passes`` /
``check_fails``) builds ``ctx.sources`` as a list of *flat* wrapped dicts
(mirroring one adapter's shape), which is why this bug shipped without an
existing regression test catching it - the harness never exercised the
nested-wrapper shape that production actually produces.
"""

from __future__ import annotations

import pytest

from dbt_bouncer.artifact_parsers.parser import wrap_dict
from dbt_bouncer.testing import check_fails, check_passes

# ---------------------------------------------------------------------------
# Hypothetical fixtures: one table, two ways ``ctx.sources`` might hand it to
# us - flat (as the unit-test harness builds it) and nested under ``.source``
# (as dbt-bouncer's own manifest parser builds it in production).
# ---------------------------------------------------------------------------

# A `sources.yml` entry (as parsed into the manifest) for a raw claims table.
# Column names are documented in lowercase snake_case, matching repo
# convention.
_HYPOTHETICAL_MANIFEST_SOURCE = {
    "columns": {
        "claim_id": {"name": "claim_id"},
        "policy_number": {"name": "policy_number"},
        "claim_status": {"name": "claim_status"},
    },
    "fqn": ["ai_file_review", "claims_raw", "claims"],
    "identifier": "CLAIMS",
    "loader": "central_ingestion",
    "name": "claims",
    "original_file_path": "models/preprocess/claims/_claims__sources.yml",
    "path": "models/preprocess/claims/_claims__sources.yml",
    "source_description": "Raw claims feed",
    "source_name": "claims_raw",
    "unique_id": "source.ai_file_review.claims_raw.claims",
}

# The catalog.json entry for the *same* table. Columns are uppercase because
# the warehouse (e.g. Snowflake) folds unquoted identifiers to uppercase.
_HYPOTHETICAL_CATALOG_NODE = {
    "columns": {
        "CLAIM_ID": {"index": 1, "name": "CLAIM_ID", "type": "TEXT"},
        "POLICY_NUMBER": {"index": 2, "name": "POLICY_NUMBER", "type": "TEXT"},
        "CLAIM_STATUS": {"index": 3, "name": "CLAIM_STATUS", "type": "TEXT"},
    },
    "metadata": {"name": "CLAIMS", "schema": "CLAIMS_RAW", "type": "BASE TABLE"},
    "unique_id": "source.ai_file_review.claims_raw.claims",
}

# The manifest source wrapped in dbt-bouncer's own container shape: the real
# source is nested one level down under `.source`, and the wrapper itself has
# no `unique_id`/`columns` of its own. This is exactly the shape
# `resource_map["sources"]` entries have in production.
_WRAPPED_MANIFEST_SOURCE = {"source": _HYPOTHETICAL_MANIFEST_SOURCE}


def _buggy_check_source_columns_are_all_documented(catalog_source, ctx_sources) -> None:
    """Pre-fix behaviour: looks up the source without unwrapping the wrapper.

    This mirrors the check as it existed before the fix. It has no fallback
    for a wrapped `ctx.sources` entry, so it blows up with an
    ``AttributeError`` instead of reporting pass/fail - a crash, not a
    governance finding.
    """
    source = next(s for s in ctx_sources if s.unique_id == catalog_source.unique_id)
    source_columns = {name.lower() for name in (source.columns or {})}
    undocumented_columns = [
        v.name
        for _, v in catalog_source.columns.items()
        if v.name.lower() not in source_columns
    ]
    assert not undocumented_columns


class TestMetadataWrappingDemo:
    def test_buggy_check_handles_flat_ctx_sources_shape(self):
        """The pre-fix logic works fine when `ctx.sources` entries are flat."""
        catalog_source = wrap_dict(_HYPOTHETICAL_CATALOG_NODE)
        ctx_sources = [wrap_dict(_HYPOTHETICAL_MANIFEST_SOURCE)]

        # No exception: this shape was never the problem.
        _buggy_check_source_columns_are_all_documented(catalog_source, ctx_sources)

    def test_buggy_check_crashes_on_wrapped_ctx_sources_shape(self):
        """The pre-fix logic crashes when `ctx.sources` entries are wrapped.

        This is the bug: dbt-bouncer doesn't fail the *check* (a governance
        finding an engineer can act on), it raises an unhandled runtime error
        and aborts the whole run. The wrapper object has no top-level
        `unique_id`, so `next(...)` never matches and instead exhausts the
        iterator, raising `StopIteration` - or, if it does happen to match
        (e.g. a case with a real top-level `unique_id`), `source.columns`
        resolves to `None` since the real columns are nested under `.source`.
        """
        catalog_source = wrap_dict(_HYPOTHETICAL_CATALOG_NODE)
        ctx_sources = [wrap_dict(_WRAPPED_MANIFEST_SOURCE)]

        with pytest.raises(StopIteration):
            _buggy_check_source_columns_are_all_documented(catalog_source, ctx_sources)

    def test_fixed_check_passes_on_flat_ctx_sources_shape(self):
        """The real, fixed check still passes for the unit-test harness shape."""
        check_passes(
            "check_source_columns_are_all_documented",
            catalog_source=_HYPOTHETICAL_CATALOG_NODE,
            ctx_sources=[_HYPOTHETICAL_MANIFEST_SOURCE],
            ctx_manifest_obj={"metadata": {"adapter_type": "snowflake"}},
        )

    def test_fixed_check_passes_on_wrapped_ctx_sources_shape(self):
        """The real, fixed check unwraps `.source` and still passes.

        Same underlying data as the crash above, but the fixed check falls
        back to unwrapping `s.source` when a `ctx.sources` entry has no
        top-level `unique_id`/`columns` of its own, so it correctly reports
        "fully documented" instead of crashing.
        """
        check_passes(
            "check_source_columns_are_all_documented",
            catalog_source=_HYPOTHETICAL_CATALOG_NODE,
            ctx_sources=[_WRAPPED_MANIFEST_SOURCE],
            ctx_manifest_obj={"metadata": {"adapter_type": "snowflake"}},
        )

    def test_fixed_check_still_fails_on_genuinely_undocumented_column(self):
        """The fix doesn't mask real findings: a truly undocumented column

        (present in the catalog node but absent from the source's `.yml`
        file) still fails the check as expected, even when `ctx.sources` is
        in the wrapped, production shape.
        """
        manifest_source_missing_status = {
            **_HYPOTHETICAL_MANIFEST_SOURCE,
            "columns": {
                "claim_id": {"name": "claim_id"},
                "policy_number": {"name": "policy_number"},
                # `claim_status` intentionally omitted to simulate drift.
            },
        }

        check_fails(
            "check_source_columns_are_all_documented",
            catalog_source=_HYPOTHETICAL_CATALOG_NODE,
            ctx_sources=[{"source": manifest_source_missing_status}],
            ctx_manifest_obj={"metadata": {"adapter_type": "snowflake"}},
        )

