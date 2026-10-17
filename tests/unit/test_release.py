"""Release metadata: version, license and source URL configuration."""

from __future__ import annotations

from pathlib import Path

from scripts.check_versions import versions
from sigma import __copyright__, __license__
from sigma.config import Config

ROOT = Path(__file__).resolve().parent.parent.parent


def test_version_is_the_same_everywhere():
    found = versions()
    assert len(set(found.values())) == 1, found


def test_license_metadata_and_files():
    assert __license__ == "AGPL-3.0-or-later"
    assert __copyright__ == "Copyright (C) 2026 Frank Miranda"
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright (C) 2026 Frank Miranda" in notice
    assert "AGPL-3.0-or-later" in notice


def test_source_url_is_configurable():
    config = Config.from_env(testing=True, source_url="https://example.test/sigma")
    assert config.source_url == "https://example.test/sigma"


def test_placeholder_secret_is_ignored():
    config = Config.from_env(testing=True, secret_key="cambia-esta-clave")
    assert config.secret_key and config.secret_key != "cambia-esta-clave"
