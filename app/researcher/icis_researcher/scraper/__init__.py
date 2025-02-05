from app.researcher.icis_researcher.scraper.beautiful_soup.beautiful_soup import BeautifulSoupScraper
from app.researcher.icis_researcher.scraper.web_base_loader.web_base_loader import WebBaseLoaderScraper
from app.researcher.icis_researcher.scraper.arxiv.arxiv import ArxivScraper
from app.researcher.icis_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper
from app.researcher.icis_researcher.scraper.browser.browser import BrowserScraper
from app.researcher.icis_researcher.scraper.tavily_extract.tavily_extract import TavilyExtract
from app.researcher.icis_researcher.scraper.scraper import Scraper

__all__ = [
    "BeautifulSoupScraper",
    "WebBaseLoaderScraper",
    "ArxivScraper",
    "PyMuPDFScraper",
    "BrowserScraper",
    "TavilyExtract",
    "Scraper"
]