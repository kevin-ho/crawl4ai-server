"""
Crawl4AI API Server — FastAPI wrapper with Firecrawl v1/v2 compat
Runs on port 11235.
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

app = FastAPI(title="Crawl4AI API", version="0.8.8")

DEFAULT_LLM_PROVIDER = "openai/crawl4ai-extract"
DEFAULT_LLM_BASE_URL = "http://100.110.38.2:4000/v1"
DEFAULT_LLM_API_TOKEN = os.environ.get("OPENAI_API_KEY", "")

HARD_MAX_PAGES = 50

# ── Persistent browser instance ──────────────────────────────────────────────
_crawler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep Playwright browser warm across requests."""
    global _crawler
    from crawl4ai import AsyncWebCrawler
    print("[INIT].... -> Starting persistent browser...")
    _crawler = AsyncWebCrawler(config=_make_browser_cfg())
    await _crawler.start()
    print("[INIT].... -> Browser ready (warm pool)")
    yield
    await _crawler.close()
    _crawler = None


app.router.lifespan_context = lifespan


# ── Request models ────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str
    onlyMainContent: Optional[bool] = True
    formats: Optional[list] = ["markdown"]


class ExtractRequest(BaseModel):
    url: str
    instruction: str
    extraction_schema: Optional[dict] = None
    chunk_token_threshold: Optional[int] = 4000
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2000


class CrawlRequest(BaseModel):
    url: str
    max_pages: Optional[int] = 10
    only_main_content: Optional[bool] = True
    same_domain_only: Optional[bool] = True


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mdgen():
    from crawl4ai import CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter
    return DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.3, threshold_type="dynamic")
    )

def _make_browser_cfg():
    from crawl4ai import BrowserConfig
    return BrowserConfig(headless=True, browser_type="chromium")

def _make_crawl_cfg(only_main=True):
    from crawl4ai import CrawlerRunConfig, CacheMode
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=_make_mdgen(),
    )

def _parse_result(result, only_main=True):
    md_obj = result.markdown
    markdown = (md_obj.fit_markdown if (only_main and md_obj.fit_markdown) else md_obj.raw_markdown)
    title = result.metadata.get("title") if isinstance(result.metadata, dict) else None
    links = result.links.get("internal", []) if isinstance(result.links, dict) else []
    return markdown, title, links


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "crawl4ai", "version": "0.8.7"}


@app.post("/scrape")
async def scrape_compat(request: Request):
    """Firecrawl/scraper compat — Hermes scraper backend."""
    from crawl4ai import CrawlerRunConfig, CacheMode

    body = await request.json()
    url = body.get("url")
    output_format = body.get("output_format", "markdown")
    start = time.time()

    try:
        result = await _crawler.arun(url=url, config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=_make_mdgen(),
        ))
        elapsed = (time.time() - start) * 1000
        if not result.success:
            return JSONResponse({"success": False, "error": result.error_message})
        markdown, _, _ = _parse_result(result)
        content = markdown
        return JSONResponse({"success": True, "content": content})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/v1/scrape")
async def scrape_v1(req: ScrapeRequest):
    """Native Crawl4AI /v1/scrape endpoint."""
    from crawl4ai import CrawlerRunConfig, CacheMode

    start = time.time()
    try:
        result = await _crawler.arun(url=req.url, config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=_make_mdgen(),
        ))
        elapsed = (time.time() - start) * 1000
        if not result.success:
            return JSONResponse({
                "success": False, "url": req.url, "status_code": result.status_code,
                "elapsed_ms": round(elapsed, 1), "error": result.error_message,
            })
        markdown, title, links = _parse_result(result, only_main=bool(req.onlyMainContent))
        return JSONResponse({
            "success": True, "url": req.url, "status_code": result.status_code,
            "elapsed_ms": round(elapsed, 1), "markdown": markdown, "title": title, "links": links,
        })
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return JSONResponse({
            "success": False, "url": req.url, "status_code": 0,
            "elapsed_ms": round(elapsed, 1), "error": str(e),
        })


@app.post("/v1/crawl")
async def crawl_v1(req: CrawlRequest):
    """Multi-page crawl — BFS link-following with a hard page limit."""
    from crawl4ai import CrawlerRunConfig, CacheMode

    start = time.time()
    max_pages = min(max(1, req.max_pages), HARD_MAX_PAGES)
    seed_url = req.url.strip()
    only_main = bool(req.only_main_content)
    same_domain = bool(req.same_domain_only)

    parsed_seed = urlparse(seed_url)
    seed_domain = parsed_seed.netloc

    seen = set()
    queue = [seed_url]
    pages = []
    errors = []

    try:
        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            try:
                result = await _crawler.arun(
                    url=url,
                    config=_make_crawl_cfg(only_main),
                )
            except Exception as e:
                errors.append({"url": url, "error": str(e)})
                continue

            if not result.success:
                errors.append({"url": url, "error": result.error_message or "scrape failed"})
                continue

            markdown, title, links = _parse_result(result, only_main=only_main)

            pages.append({
                "url": url,
                "title": title or "",
                "markdown": markdown or "",
                "status_code": result.status_code,
            })

            # Extract links to follow (links are dicts with 'href' key)
            if len(pages) < max_pages:
                for link in links:
                    href = link if isinstance(link, str) else link.get("href", "")
                    if not href or href in seen:
                        continue
                    parsed = urlparse(href)
                    # Only follow http/https links
                    if parsed.scheme not in ("http", "https"):
                        continue
                    # Same-domain filter
                    if same_domain and parsed.netloc != seed_domain:
                        continue
                    queue.append(href)

        elapsed = (time.time() - start) * 1000
        return JSONResponse({
            "success": True,
            "url": seed_url,
            "pages_found": len(seen),
            "pages_returned": len(pages),
            "max_pages": max_pages,
            "elapsed_ms": round(elapsed, 1),
            "pages": pages,
            "errors": errors if errors else None,
        })

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return JSONResponse({
            "success": False,
            "url": seed_url,
            "elapsed_ms": round(elapsed, 1),
            "error": str(e),
        })


@app.post("/v1/extract")
async def extract(req: ExtractRequest):
    from crawl4ai import CrawlerRunConfig, CacheMode, LLMConfig, LLMExtractionStrategy

    start = time.time()
    try:
        llm_config = LLMConfig(
            provider=DEFAULT_LLM_PROVIDER,
            api_token=DEFAULT_LLM_API_TOKEN,
            base_url=DEFAULT_LLM_BASE_URL,
        )
        extra_args = {"temperature": req.temperature, "max_tokens": req.max_tokens}

        if req.extraction_schema:
            llm_strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                schema=req.extraction_schema,
                extraction_type="schema",
                instruction=req.instruction,
                chunk_token_threshold=req.chunk_token_threshold,
                overlap_rate=0.0, apply_chunking=True,
                input_format="markdown", extra_args=extra_args, verbose=False,
            )
        else:
            llm_strategy = LLMExtractionStrategy(
                llm_config=llm_config,
                extraction_type="block",
                instruction=req.instruction,
                chunk_token_threshold=req.chunk_token_threshold,
                overlap_rate=0.0, apply_chunking=True,
                input_format="markdown", extra_args=extra_args, verbose=False,
            )

        result = await _crawler.arun(url=req.url, config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=_make_mdgen(),
            extraction_strategy=llm_strategy,
        ))

        elapsed = (time.time() - start) * 1000
        if not result.success:
            return JSONResponse({
                "success": False, "url": req.url, "elapsed_ms": round(elapsed, 1),
                "error": result.error_message,
            })

        extracted = None
        if result.extracted_content:
            try:
                extracted = __import__("json").loads(result.extracted_content)
            except Exception:
                extracted = result.extracted_content

        return JSONResponse({
            "success": True, "url": req.url, "elapsed_ms": round(elapsed, 1),
            "extracted": extracted, "llm_provider": DEFAULT_LLM_PROVIDER,
        })

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return JSONResponse({
            "success": False, "url": req.url, "elapsed_ms": round(elapsed, 1), "error": str(e),
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11235, workers=1)
