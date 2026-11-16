"""Smoke tests validating the package scaffold."""

import clinverify


def test_version_is_defined() -> None:
    assert clinverify.__version__ == "0.1.0"


def test_public_modules_import() -> None:
    from clinverify import api, generator, ingestion, retrieval

    assert api is not None
    assert generator is not None
    assert ingestion is not None
    assert retrieval is not None
