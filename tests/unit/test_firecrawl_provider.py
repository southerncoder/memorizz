"""Unit tests for the Firecrawl internet provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memorizz.internet_access.providers.firecrawl import FirecrawlProvider


@pytest.mark.unit
def test_firecrawl_truncates_large_pages():
    provider = FirecrawlProvider(
        api_key="test-key",
        base_url="https://example.com",
        config={"max_content_chars": 20},
    )
    provider._post = MagicMock(
        return_value={
            "markdown": "A" * 50,
            "metadata": {"title": "Example"},
        }
    )

    page = provider.fetch_url("https://memorizz.ai")

    assert page.metadata["content_truncated"] is True
    assert page.metadata["content_original_characters"] == 50
    assert page.metadata["content_returned_characters"] == 20
    assert page.content.startswith("A" * 20)
    assert "trimmed the page" in page.content
    assert page.raw is None


@pytest.mark.unit
def test_firecrawl_can_include_sanitized_raw_payload():
    provider = FirecrawlProvider(
        api_key="test-key",
        base_url="https://example.com",
        config={"include_raw_response": True, "max_raw_chars": 5},
    )
    provider._post = MagicMock(
        return_value={
            "markdown": "abcdefg",
            "metadata": {"title": "Example"},
            "nested": {"rawHtml": "<p>" + "x" * 20},
        }
    )

    page = provider.fetch_url("https://memorizz.ai")

    assert page.raw is not None
    assert page.raw["markdown"].startswith("abcde")
    assert "truncated" in page.raw["markdown"]
    assert "truncated" in page.raw["nested"]["rawHtml"]
