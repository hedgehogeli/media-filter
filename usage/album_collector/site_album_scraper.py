import re
import os
import random
import asyncio
import logging
from urllib.parse import urlparse, urljoin
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import multiprocessing
import html

# Import the client from your existing module
from deployment.client import MLBatchClient
from deployment.common import ImageScore
from site_utils.image_processing import ImageProcessor, MAX_SIZE_KB
from site_utils.playwright_scraper import PlaywrightHandler, HEADERS
from site_utils.custom_regex import RegexFilter

# Configuration
SHOW_BAD_ALBUMS = True
MAX_PAGES = 1000

BATCH_SIZE = 20
RESULT_HTML_PATH = "result.html"


class AlbumRating(Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    NEUTRAL = "NEUTRAL"


@dataclass
class AlbumInfo:
    """Represents album metadata"""

    url: str
    title: str
    date: str
    size_text: str
    num_files: int
    total_size_mb: float
    avg_file_size_mb: float

    @property
    def total_size_gb(self) -> float:
        return self.total_size_mb / 1024


@dataclass
class PreviewImage:
    """Represents a preview image ready for ML processing"""

    filename: str
    data: bytes
    source_type: str  # 'image' or 'video'


@dataclass
class ProcessedAlbum:
    """Represents a fully processed album with ML results"""

    info: AlbumInfo
    rating: AlbumRating
    regex_match: str = ""
    ml_scores: List[ImageScore] = field(default_factory=list)
    avg_scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    processing_time: float = 0.0
    error: Optional[str] = None


class SiteScraper:
    """Main scraper class for Site albums"""

    def __init__(self, ml_client: MLBatchClient, regex_filter: RegexFilter):
        self.ml_client = ml_client
        self.regex_filter = regex_filter
        self.logger = logging.getLogger(__name__)
        self.compressor = ImageProcessor()

    async def fetch_page_content(self, url: str) -> str:
        """Fetch HTML content using Playwright"""
        return await PlaywrightHandler.fetch_page_content(url)

    def truncate_html(self, html: str) -> str:
        """Truncate HTML at player script marker"""
        marker = '<script src="../js/player.js"></script>'
        if marker in html:
            return html.split(marker)[0]
        return html

    def parse_album_info(self, album_url: str, html: str) -> Optional[AlbumInfo]:
        """Extract album metadata from HTML"""
        try:
            # Extract title
            title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "Unknown"

            # Extract date
            date_match = re.search(r"([0-9]{2}/[0-9]{2}/[0-9]{4})", html)
            date = date_match.group() if date_match else "??/??/????"

            # Extract size info
            size_match = re.search(r"<span[^>]*>(.*?iles.*?)</span>", html)
            size_text = size_match.group(1).strip() if size_match else "? files, ? MB"

            # Parse size details
            num_files = 0
            total_size_mb = 0.0

            # Extract number of files
            files_match = re.search(r"(\d+)\s*files?", size_text, re.IGNORECASE)
            if files_match:
                num_files = int(files_match.group(1))

            # Extract size
            size_match = re.search(r"([\d.]+)\s*(GB|MB|KB)", size_text, re.IGNORECASE)
            if size_match:
                size_val = float(size_match.group(1))
                unit = size_match.group(2).upper()
                if unit == "GB":
                    total_size_mb = size_val * 1024
                elif unit == "MB":
                    total_size_mb = size_val
                elif unit == "KB":
                    total_size_mb = size_val / 1024

            avg_file_size_mb = total_size_mb / num_files if num_files > 0 else 0

            return AlbumInfo(
                url=album_url,
                title=title,
                date=date,
                size_text=size_text,
                num_files=num_files,
                total_size_mb=total_size_mb,
                avg_file_size_mb=avg_file_size_mb,
            )
        except Exception as e:
            self.logger.error(f"Error parsing album info: {e}")
            return None

    def get_file_links(self, album_url: str, html: str) -> List[str]:
        """Extract file page links from album HTML"""
        raw_links = re.findall(r'href="([^\"]+)"', html)
        parsed = urlparse(album_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        file_links = []
        seen = set()
        for link in raw_links:
            if link.startswith("/f/"):
                full = urljoin(base, link)
                if full not in seen:
                    seen.add(full)
                    file_links.append(full)

        return file_links

    async def get_file_preview(
        self, file_url: str, album_url: str
    ) -> Optional[PreviewImage]:
        """Extract and create preview from file page"""
        try:
            html = await self.fetch_page_content(file_url)
            html = self.truncate_html(html)

            # Check for video cover URL
            script_match = re.search(
                r'var\s+videoCoverUrl\s*=\s*"([^\"]+_grid\.png)"', html
            )
            if script_match:
                cover_url = script_match.group(1).replace("\\/", "/")
                # Download and crop video thumbnail
                resp = requests.get(
                    cover_url, headers={**HEADERS, "Referer": file_url}, timeout=20
                )
                resp.raise_for_status()

                # Crop top-right quarter
                img = Image.open(BytesIO(resp.content))
                width, height = img.size
                crop_box = (width * 3 // 4, 0, width, height // 4)
                cropped = img.crop(crop_box)

                # Convert to bytes
                output = BytesIO()
                cropped.save(output, format="PNG")
                preview_data = output.getvalue()

                # Compress if needed
                if len(preview_data) > MAX_SIZE_KB * 1024:
                    preview_data = self.compressor.compress(preview_data)

                return PreviewImage(
                    filename=os.path.basename(file_url),
                    data=preview_data,
                    source_type="video",
                )

            # Check for direct image
            img_match = re.search(r'<img[^>]+src="([^\"]*site[^\"]+)"', html)
            if img_match:
                img_url = img_match.group(1)
                resp = requests.get(
                    img_url, headers={**HEADERS, "Referer": file_url}, timeout=20
                )
                resp.raise_for_status()

                # Compress image
                compressed = self.compressor.compress(resp.content)

                return PreviewImage(
                    filename=os.path.basename(file_url),
                    data=compressed,
                    source_type="image",
                )

            return None

        except Exception as e:
            self.logger.error(
                f"Error getting file preview - Album: {album_url}, File: {file_url}, Error: {e}"
            )
            return None

    def predict_from_softmax(self, scores: List[float]) -> AlbumRating:
        """Convert ML scores to album rating"""
        if scores[0] > 0.5:  # Assuming first score indicates "bad"
            return AlbumRating.BAD
        elif scores[2] > 0.1:  # Assuming third score indicates "good"
            return AlbumRating.GOOD
        else:
            return AlbumRating.NEUTRAL

    async def process_album(self, album_url: str) -> Optional[ProcessedAlbum]:
        """Process a single album completely"""
        self.logger.info(f"Starting processing album: {album_url}")
        try:
            # Fetch album page
            html = await self.fetch_page_content(album_url)

            # Parse album info
            album_info = self.parse_album_info(album_url, html)
            if not album_info:
                return None

            # Check regex filters
            accept, regex_match = self.regex_filter.matches(html)
            if not accept:
                # Album blacklisted
                return None

            # Get file links
            file_links = self.get_file_links(album_url, html)
            if not file_links:
                return ProcessedAlbum(
                    info=album_info, rating=AlbumRating.NEUTRAL, regex_match=regex_match
                )

            # Randomly sample up to 10 files
            sample_size = min(10, len(file_links))
            sampled_files = random.sample(file_links, sample_size)

            # Get previews
            previews = []
            for file_url in sampled_files:
                preview = await self.get_file_preview(file_url, album_url)
                if preview:
                    previews.append(preview)

            if not previews:
                return ProcessedAlbum(
                    info=album_info,
                    rating=AlbumRating.NEUTRAL,
                    regex_match=regex_match,
                    avg_scores=[0.0, 0.0, 0.0],
                )

            # Send to ML model
            batch_data = [(p.filename, p.data) for p in previews]
            response = self.ml_client.submit_batch_sync(batch_data, timeout=30.0)

            # Aggregate scores and determine rating
            avg_scores = [0.0, 0.0, 0.0]
            valid_count = 0

            for score in response.scores:
                if not score.error:
                    avg_scores[0] += score.score1
                    avg_scores[1] += score.score2
                    avg_scores[2] += score.score3
                    valid_count += 1

            if valid_count > 0:
                avg_scores = [s / valid_count for s in avg_scores]
                rating = self.predict_from_softmax(avg_scores)
            else:
                rating = AlbumRating.NEUTRAL

            return ProcessedAlbum(
                info=album_info,
                rating=rating,
                regex_match=regex_match,
                ml_scores=response.scores,
                avg_scores=avg_scores,
                processing_time=response.processing_time,
            )

        except Exception as e:
            self.logger.error(f"Error processing album {album_url}: {e}")
            return None


class HTMLReporter:
    """Handles HTML output generation"""

    def __init__(
        self, output_path: str = RESULT_HTML_PATH, error_path: str = "error.html"
    ):
        self.output_path = output_path
        self.error_path = error_path
        self._ensure_html_exists()
        self._ensure_error_html_exists()

    def _ensure_html_exists(self):
        """Create HTML file with header if doesn't exist"""
        if not os.path.exists(self.output_path):
            header = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            background-color: #293134;
            font-family: Consolas, monospace;
            font-size: 16px;
            color: #e0e2e4;
        }
        p { margin: 5px 0; }
        a:link { color: #9496eb; text-decoration: none; }
        a:visited { color: #d67e8e; text-decoration: none; }
        .grey { color: #808080; }
        .orange { color: #FFA500; }
        .red { color: #FF0000; }
        .green { color: #00FF00; }
        .match { color: #90EE90; font-weight: bold; }
        .scores { color: #87CEEB; font-family: monospace; }
    </style>
</head>
<body>
"""
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(header)

    def _ensure_error_html_exists(self):
        """Create error HTML file if doesn't exist"""
        if not os.path.exists(self.error_path):
            header = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            background-color: #293134;
            font-family: Consolas, monospace;
            font-size: 16px;
            color: #e0e2e4;
        }
        p { margin: 5px 0; }
        a:link { color: #FF6B6B; text-decoration: none; }
        a:visited { color: #CC5555; text-decoration: none; }
    </style>
</head>
<body>
<p>Albums that failed to process:</p>
"""
            with open(self.error_path, "w", encoding="utf-8") as f:
                f.write(header)

    def find_last_links(self, num_links: int = 50) -> Set[str]:
        """Extract last N album URLs from HTML"""
        if not os.path.exists(self.output_path):
            return set()

        with open(self.output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all album links
        links = re.findall(r'href="(https://[^"]+/a/[^"]+)"', content)
        return set(links[-num_links:]) if links else set()

    def append_album(self, album: ProcessedAlbum):
        """Append processed album to HTML"""
        # Skip BAD albums unless SHOW_BAD_ALBUMS is True
        if album.rating == AlbumRating.BAD and not SHOW_BAD_ALBUMS:
            return

        # Build the HTML line
        parts = []

        # Add ML scores at the beginning
        scores_str = f"[{album.avg_scores[0]:.2f}, {album.avg_scores[1]:.2f}, {album.avg_scores[2]:.2f}]"
        parts.append(f'<span class="scores">{scores_str}</span>')

        # Regex match or date
        if album.regex_match:
            parts.append(f'<span class="match">{html.escape(album.regex_match)}</span>')
        parts.append(album.info.date)

        # Album link and title
        title_class = "grey" if album.rating == AlbumRating.BAD else ""
        title_html = (
            f'<span class="{title_class}">{html.escape(album.info.title)}</span>'
            if title_class
            else html.escape(album.info.title)
        )

        # Parse size text properly
        size_parts = album.info.size_text.split(",")
        size_str = (
            size_parts[1].strip() if len(size_parts) > 1 else album.info.size_text
        )

        # Size coloring
        size_class = ""
        if album.info.total_size_gb > 10:
            size_class = "red"
        elif album.info.total_size_gb > 5:
            size_class = "orange"
        size_html = (
            f'<span class="{size_class}">{html.escape(size_str)}</span>'
            if size_class
            else html.escape(size_str)
        )

        # File count coloring
        files_class = ""
        if album.info.num_files > 100:
            files_class = "green"
        elif album.info.avg_file_size_mb > 1024:
            files_class = "red"
        files_html = (
            f'<span class="{files_class}">{album.info.num_files} files</span>'
            if files_class
            else f"{album.info.num_files} files"
        )

        # Construct line
        line = f'<p>{" ".join(parts)} <a href="{html.escape(album.info.url)}" target="_blank">{title_html} [{size_html}, {files_html}]</a></p>\n'

        # Append to file
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(line)

    def append_error(self, album_url: str, error: str):
        """Append failed album to error HTML"""
        line = f'<p><a href="{html.escape(album_url)}" target="_blank">{html.escape(album_url)}</a> - {html.escape(error)}</p>\n'
        with open(self.error_path, "a", encoding="utf-8") as f:
            f.write(line)

    def add_processing_start(self):
        """Add processing start timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"<p>Starting processing: {timestamp}</p>\n"
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(line)


class AlbumCollector:
    """Collects album URLs from Site"""

    def __init__(self, max_pages: int = MAX_PAGES):
        self.max_pages = max_pages
        self.logger = logging.getLogger(__name__)

    async def collect_album_urls(
        self, existing_urls: Set[str], is_first_run: bool = False
    ) -> List[str]:
        from playwright.async_api import async_playwright

        """Collect album URLs until duplicates found or max pages reached"""
        album_urls = []
        seen_urls = set(existing_urls)
        duplicate_count = 0

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers(HEADERS)

            for page_num in range(1, self.max_pages + 1):
                url = f"https://www.reddit.com/?search=&page={page_num}"
                await page.goto(url)
                links = await page.eval_on_selector_all(
                    "main a", "elements => elements.map(e => e.href)"
                )

                for link in links:
                    if not link:
                        continue

                    album_id = link.rstrip("/").split("/")[-1]
                    if len(album_id) != 8:
                        continue

                    if link in existing_urls:
                        duplicate_count += 1
                        self.logger.info(f"Found duplicate: {link}")
                        if duplicate_count >= 5:
                            break
                    elif link not in seen_urls:
                        album_urls.append(link)
                        seen_urls.add(link)

                if duplicate_count >= 5:
                    break

                # On first run, limit to reasonable number of pages
                if is_first_run and page_num >= 30:
                    self.logger.info(f"First run: limiting to {page_num} pages")
                    break

            await browser.close()

        self.logger.info(f"Collected {len(album_urls)} albums from {page_num} pages")
        return album_urls


async def process_album_with_retry(
    scraper: SiteScraper, album_url: str, max_retries: int = 3
) -> Tuple[Optional[ProcessedAlbum], Optional[Tuple[str, str]]]:
    """Process album with retry logic"""
    retry_count = 0
    backoff_seconds = 1

    while retry_count < max_retries:
        try:
            result = await scraper.process_album(album_url)
            return (result, None)
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                logging.warning(
                    f"Retry {retry_count}/{max_retries} for {album_url} after {backoff_seconds}s"
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds *= 2
            else:
                logging.error(
                    f"Failed to process {album_url} after {max_retries} attempts: {e}"
                )
                return (None, (album_url, str(e)))

    return (None, (album_url, "Max retries exceeded"))


async def main():
    """Main processing function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize components
    ml_client = MLBatchClient(server_host="192.168.56.1", server_port=5000)
    regex_filter = RegexFilter()
    scraper = SiteScraper(ml_client, regex_filter)
    reporter = HTMLReporter()
    collector = AlbumCollector()

    try:
        # Connect to ML server
        ml_client.connect()

        # Get existing URLs from HTML
        existing_urls = reporter.find_last_links()
        is_first_run = len(existing_urls) == 0

        # Add processing start marker
        reporter.add_processing_start()

        # Collect new album URLs
        album_urls = await collector.collect_album_urls(existing_urls, is_first_run)
        logging.info(f"Found {len(album_urls)} new albums to process")
        album_urls.reverse()

        # Process albums in parallel batches
        batch_size = min(
            multiprocessing.cpu_count() * 2, 10
        )  # Limit concurrent requests
        all_results = []
        failed_albums = []

        for i in range(0, len(album_urls), batch_size):
            batch = album_urls[i : i + batch_size]

            # Process batch in parallel
            batch_tasks = [process_album_with_retry(scraper, url) for url in batch]
            batch_results = await asyncio.gather(*batch_tasks)

            # Separate successful and failed results
            for result, error in batch_results:
                if result:
                    all_results.append(result)
                elif error:
                    failed_albums.append(error)

            # Write to HTML every 10 albums
            if len(all_results) >= 10:
                # Sort and write the oldest 10 albums
                all_results.sort(key=lambda x: x.info.date)
                to_write = all_results[:10]
                all_results = all_results[10:]

                for album in to_write:
                    reporter.append_album(album)
                logging.info(
                    f"Written {(i // batch_size + 1) * batch_size} albums to HTML"
                )

        # Write remaining results
        if all_results:
            all_results.sort(key=lambda x: x.info.date)
            for album in all_results:
                reporter.append_album(album)

        # Write failed albums to error.html
        for album_url, error in failed_albums:
            reporter.append_error(album_url, error)

        total_processed = len(album_urls) - len(failed_albums)
        logging.info(
            f"Processing completed. Processed: {total_processed}, Failed: {len(failed_albums)}"
        )

    except Exception as e:
        logging.error(f"Main processing error: {e}")
        raise
    finally:
        ml_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
