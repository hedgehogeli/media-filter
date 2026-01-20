import asyncio

from playwright.async_api import async_playwright

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    "Referer": "",
}


class PlaywrightHandler:
    @staticmethod
    async def _fetch_content(url: str, scroll_pause_time: float = 2.0) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers(HEADERS)
            await page.goto(url, wait_until="networkidle", timeout=20000)

            # Get initial page height
            last_height = await page.evaluate("document.body.scrollHeight")

            while True:
                # Scroll to bottom
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

                # Wait for new content to load
                await page.wait_for_timeout(int(scroll_pause_time * 1000))

                # Calculate new scroll height
                new_height = await page.evaluate("document.body.scrollHeight")

                # Break if no new content loaded
                if new_height == last_height:
                    break

                last_height = new_height

            # Get final content
            content = await page.content()
            await browser.close()
            return content

    @staticmethod
    async def _fetch_content_with_element_check(
        url: str, target_selector: str = None, max_scrolls: int = 10
    ) -> str:
        """
        Alternative method that scrolls until a specific element appears or max scrolls reached
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers(HEADERS)
            await page.goto(url, wait_until="networkidle")

            scrolls = 0
            while scrolls < max_scrolls:
                # Check if target element exists (if specified)
                if target_selector:
                    element = await page.query_selector(target_selector)
                    if element:
                        break

                # Get current position
                prev_position = await page.evaluate("window.pageYOffset")

                # Scroll down
                await page.evaluate("window.scrollBy(0, window.innerHeight)")

                # Wait for content to load
                await page.wait_for_timeout(2000)

                # Check if we've actually scrolled
                curr_position = await page.evaluate("window.pageYOffset")
                if prev_position == curr_position:
                    # We've reached the bottom
                    break

                scrolls += 1

            content = await page.content()
            await browser.close()
            return content

    @staticmethod
    async def _fetch_content_incremental(
        url: str, scroll_pause_time: float = 1.0
    ) -> str:
        """
        Scrolls incrementally in smaller chunks for better compatibility
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers(HEADERS)
            await page.goto(url, wait_until="networkidle")

            # Get viewport and document heights
            viewport_height = await page.evaluate("window.innerHeight")
            last_height = await page.evaluate("document.body.scrollHeight")

            current_position = 0

            while current_position < last_height:
                # Scroll by viewport height
                current_position += viewport_height
                await page.evaluate(f"window.scrollTo(0, {current_position})")

                # Wait for content to load
                await page.wait_for_timeout(int(scroll_pause_time * 1000))

                # Update document height
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height > last_height:
                    last_height = new_height

            # Final scroll to absolute bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(int(scroll_pause_time * 1000))

            content = await page.content()
            await browser.close()
            return content

    @staticmethod
    async def fetch_page_content(url: str, method: str = "scroll_to_bottom", **kwargs) -> str:
        """
        Fetch HTML content of a URL using Playwright with scroll support.

        Args:
            url: The URL to fetch
            method: The scrolling method to use:
                - 'scroll_to_bottom': Scrolls to bottom until no new content loads
                - 'element_check': Scrolls until a specific element appears
                - 'incremental': Scrolls in viewport-sized chunks
            **kwargs: Additional arguments passed to the specific method

        Returns:
            The full HTML content after scrolling
        """
        if method == "scroll_to_bottom":
            return await PlaywrightHandler._fetch_content(url, **kwargs)
        elif method == "element_check":
            return await PlaywrightHandler._fetch_content_with_element_check(url, **kwargs)
        elif method == "incremental":
            return await PlaywrightHandler._fetch_content_incremental(url, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
