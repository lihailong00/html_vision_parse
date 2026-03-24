"""
Standard test URLs for benchmarking.

Each URL represents a different type of page to test different scenarios.
"""

# Short pages - news homepages, landing pages
SHORT_PAGES = [
    "https://news.ycombinator.com",
    "https://www.bbc.com/news/technology",
    "https://www.minimaxi.com/models/text/m27",
]

# Long pages - articles, documentation
LONG_PAGES = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.zhihu.com/question/19562333",
]

# Complex pages - e-commerce, forums
COMPLEX_PAGES = [
    "https://news.ycombinator.com",
    "https://www.amazon.com/s?k=laptop",
]

# All test URLs
BENCHMARK_URLS = {
    "short": SHORT_PAGES,
    "long": LONG_PAGES,
    "complex": COMPLEX_PAGES,
    "all": SHORT_PAGES + LONG_PAGES + COMPLEX_PAGES,
}


if __name__ == "__main__":
    print("Benchmark URLs:")
    for category, urls in BENCHMARK_URLS.items():
        print(f"\n{category.upper()}:")
        for url in urls:
            print(f"  - {url}")

