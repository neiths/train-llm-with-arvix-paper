# Scrapy settings for arxiv_crawler project

BOT_NAME = "arxiv_crawler"

SPIDER_MODULES = ["src.collectors.scrapy_crawler.spiders"]
NEWSPIDER_MODULE = "src.collectors.scrapy_crawler.spiders"

# Crawl responsibly by identifying yourself
USER_AGENT = "ArxivLLMBot/1.0 (+https://github.com/thienhb/train-llm-with-arxiv-data)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Configure delays between requests
DOWNLOAD_DELAY = 0.5
RANDOMIZE_DOWNLOAD_DELAY = True

# Disable cookies
COOKIES_ENABLED = False

# Configure retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Configure item pipelines
ITEM_PIPELINES = {
    "src.collectors.scrapy_crawler.pipelines.ValidationPipeline": 100,
    "src.collectors.scrapy_crawler.pipelines.StoragePipeline": 200,
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 1 day
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Configure logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# Request fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"

# Twisted reactor
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Feed exports
FEED_EXPORT_ENCODING = "utf-8"
