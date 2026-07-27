from __future__ import annotations

import tempfile

from deltasub.gains.fixture import run_fixture_collection


def test_m4_fixture_cache_resume_is_idempotent():
    with tempfile.TemporaryDirectory() as directory:
        first = run_fixture_collection(directory)
        resumed = run_fixture_collection(directory, resume=True)
        assert resumed["cache_id"] == first["cache_id"]
        assert resumed["processed_candidates"] == 0
        assert resumed["skipped_existing_candidates"] == first["record_count"]
        assert resumed["shard_sha256_values"] == first["shard_sha256_values"]
        assert resumed["deterministic_validation_hash"] == first["deterministic_validation_hash"]
