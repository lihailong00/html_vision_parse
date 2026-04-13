"""Tests for SimplePipeline."""

import pytest
from src.simple_pipeline import (
    SimplePipeline,
    ExtractionResult,
    extract_country_from_url,
    extract_city_from_url,
    extract_lang_from_tld,
)


class TestURLParsing:
    """Test URL parsing helpers."""

    def test_extract_country_from_url_cn(self):
        assert extract_country_from_url("https://example.cn/article") == "CN"

    def test_extract_country_from_url_jp(self):
        assert extract_country_from_url("https://news.yahoo.co.jp/politics") == "JP"

    def test_extract_country_from_url_generic(self):
        # Generic TLDs should return None
        assert extract_country_from_url("https://example.com/article") is None
        assert extract_country_from_url("https://example.org/article") is None

    def test_extract_country_from_url_with_www(self):
        assert extract_country_from_url("https://www.example.cn/article") == "CN"

    def test_extract_city_from_url_beijing(self):
        city = extract_city_from_url("https://beijing.example.com/article")
        assert city == "Beijing"

    def test_extract_city_from_url_shanghai(self):
        city = extract_city_from_url("https://shanghai.xxx.com/article")
        assert city == "Shanghai"

    def test_extract_city_from_url_with_www(self):
        # www should be filtered out
        city = extract_city_from_url("https://www.example.com/article")
        assert city is None

    def test_extract_lang_from_tld(self):
        assert extract_lang_from_tld("https://example.cn/article") == "zh"
        assert extract_lang_from_tld("https://example.jp/article") == "ja"
        assert extract_lang_from_tld("https://example.kr/article") == "ko"

    def test_extract_lang_from_generic_tld(self):
        # Generic TLDs don't imply language
        assert extract_lang_from_tld("https://example.com/article") is None


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_to_dict_all_fields(self):
        result = ExtractionResult(
            title="Test Title",
            content="Test Content",
            publish_time="2026-04-13 10:00",
            lang_type="en",
            country="US",
            city="NYC",
        )
        d = result.to_dict()
        assert d["title"] == "Test Title"
        assert d["content"] == "Test Content"
        assert d["publish_time"] == "2026-04-13 10:00"
        assert d["lang_type"] == "en"
        assert d["country"] == "US"
        assert d["city"] == "NYC"

    def test_to_dict_with_nulls(self):
        result = ExtractionResult()
        d = result.to_dict()
        assert all(v is None for v in d.values())

    def test_to_dict_with_error(self):
        result = ExtractionResult(error="Test error", raw_response="raw")
        d = result.to_dict()
        assert d["error"] == "Test error"
        assert d["raw_response"] == "raw"
