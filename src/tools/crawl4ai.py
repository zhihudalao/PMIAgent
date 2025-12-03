from src.config.logger import get_logger


from langchain_core.tools import tool
from crawl4ai import AsyncWebCrawler

from .decorators import log_io

logger = get_logger(__name__)


# 定义核心异步爬虫函数
async def quick_crawl_tool(urls: list[str]) -> list[dict]:
    """异步爬取多个URL并返回markdown内容"""
    results = []
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                result = await crawler.arun(url=url)
                if result.success:
                    results.append({
                        "url": url,
                        "content": result.markdown,
                        "title": result.metadata.get("title", ""),
                        "status": "success"
                    })
                else:
                    results.append({
                        "url": url,
                        "content": "",
                        "error": "爬取失败",
                        "status": "failed"
                    })
            except Exception as e:
                results.append({
                    "url": url,
                    "content": "",
                    "error": str(e),
                    "status": "error"
                })
    return results


# 封装成LangChain Tool
@tool
@log_io
async def crawl4ai_tool(urls: list[str]) -> dict:
    """用于爬取网页内容。接收URL列表，返回对应网页的内容。

    参数:
        urls: URL字符串列表，例如 ["https://example.com"]
    """
    print(f'crawl4ai_tool收到的完整输入: {urls}\n')
    result = await quick_crawl_tool(urls)
    return {"result": result}



from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import CrawlerRunConfig
from pydantic import BaseModel

class ArticleSummary(BaseModel):
    title: str
    key_points: list[str]
    summary: str  # 200字内摘要
    data_points: list[str]

config = CrawlerRunConfig(
    extraction_strategy=LLMExtractionStrategy(
        provider="openai/gpt-4",
        api_token="sk-...",
        schema=ArticleSummary.model_json_schema(),
        instruction="提取文章核心信息，忽略广告和导航内容",
    )
)
