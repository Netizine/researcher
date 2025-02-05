from typing import List, Dict
from app.researcher.state import AgentState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from app.researcher.icis_researcher.actions.utils import stream_output
from app.researcher.icis_researcher.actions.web_scraping import scrape_urls
from app.researcher.icis_researcher.scraper.utils import get_image_hash


class BrowserSkill:
    def __init__(self, state: AgentState, config: RunnableConfig):
        self.state = state
        self.config = config

    async def browse_urls(self, urls: List[str]) -> List[Dict]:
        """Browse a list of URLs and extract their content"""
        try:
            self.state["research_logs"].append({
                "message": f"Browsing {len(urls)} URLs",
                "done": False
            })
            await copilotkit_emit_state(self.config, self.state)

            # Extract content from URLs
            scraped_content, images = await self._scrape_urls(urls)

            self.state["research_logs"].append({
                "message": f"Successfully extracted content from {len(urls)} URLs",
                "done": True
            })
            await copilotkit_emit_state(self.config, self.state)

            # Add research sources and images to state
            self.state["research_sources"] = scraped_content
            self.state["research_images"] = self.select_top_images(images, k=4)

            return scraped_content

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error browsing URLs: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    async def _scrape_urls(self, urls: List[str]) -> (List[Dict], List[Dict]):
        """Scrape content from a list of URLs"""
        try:
            # Scrape content using existing implementation
            scraped_content, images = scrape_urls(urls, self.config)

            return scraped_content, images

        except Exception as e:
            self.state["research_logs"].append({
                "message": f"Error scraping URLs: {str(e)}",
                "done": True,
                "error": True
            })
            await copilotkit_emit_state(self.config, self.state)
            raise e

    def select_top_images(self, images: List[Dict], k: int = 2) -> List[str]:
        """
        Select most relevant images and remove duplicates based on image content.

        Args:
            images (List[Dict]): List of image dictionaries with 'url' and 'score' keys.
            k (int): Number of top images to select if no high-score images are found.

        Returns:
            List[str]: List of selected image URLs.
        """
        unique_images = []
        seen_hashes = set()
        current_research_images = self.state.get("research_images", [])

        # First, select all score 2 and 3 images
        high_score_images = [img for img in images if img['score'] >= 2]

        for img in high_score_images + images:  # Process high-score images first, then all images
            img_hash = get_image_hash(img['url'])
            if img_hash and img_hash not in seen_hashes and img['url'] not in current_research_images:
                seen_hashes.add(img_hash)
                unique_images.append(img['url'])

                if len(unique_images) == k:
                    break

        return unique_images
