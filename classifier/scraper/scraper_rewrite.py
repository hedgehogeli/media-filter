import asyncio
import logging
import multiprocessing as mp
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
from lxml import html
from PIL import Image
from playwright.async_api import async_playwright


@dataclass
class AlbumResult:
    album_url: str
    album_id: str
    num_previews: int
    num_files: int
    num_total: int
    incomplete: bool


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    ),
    "Referer": "",
}

MAX_SIZE_KB = 500
MAX_DIMENSION = 1920

SUFFIX_PREVIEW = "_preview"
SUFFIX_COMPRESSED = "_cmpr_img"
SUFFIX_ORIGINAL = "_orig_img"
SUFFIX_VIDEO_TR = "_vid_screen_tr"
SUFFIX_VIDEO_BL = "_vid_screen_bl"

# file extensions we can't deal with, and don't want to count towards our count of valid files
# fmt: off
UNDESIREABLE_EXTENSIONS = {
    ".mkv", ".ts",
    ".svg",
    ".pdf", ".doc", ".docx", ".epub", ".txt",
    ".mp3", ".m4a", ".wav", ".opus", ".flac",
    ".7z", ".gz", ".bz2", ".xz", "rar", ".zip",
    ".001", ".002", ".003", ".004", ".005"
}
# fmt: on


class ImageProcessor:
    @staticmethod
    def get_image_dimensions_from_bytes(
        image_bytes: bytes,
    ) -> tuple[Optional[int], Optional[int]]:
        """Returns (width, height) using ffprobe on image bytes via pipe."""
        try:
            process = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    "-f",
                    "image2pipe",  # Read from pipe
                    "-i",
                    "pipe:0",  # Input from stdin
                ],
                input=image_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Removed text=True to handle binary input
            )

            # Decode the output from bytes to string
            output = process.stdout.decode("utf-8").strip()

            if process.returncode != 0 or not output:
                return None, None

            width, height = map(int, output.split("x"))
            return width, height
        except:
            return None, None

    @staticmethod
    def compress_image_in_memory(
        image_bytes: bytes, target_size_kb: int = MAX_SIZE_KB
    ) -> bytes:
        """Compress image bytes using ffmpeg entirely in memory, returning compressed bytes."""
        width, height = ImageProcessor.get_image_dimensions_from_bytes(image_bytes)
        scale_filter = ""

        if width and height and (width > MAX_DIMENSION or height > MAX_DIMENSION):
            if width > height:
                scale_filter = f"scale={MAX_DIMENSION}:-2"
            else:
                scale_filter = f"scale=-2:{MAX_DIMENSION}"

        quality = 4
        max_attempts = 10
        best_result = image_bytes
        best_size_diff = float("inf")

        for _ in range(max_attempts):
            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "image2pipe",  # Input format from pipe
                "-i",
                "pipe:0",  # Read from stdin
            ]

            # Add scaling filter if needed
            if scale_filter:
                cmd.extend(["-vf", scale_filter])

            cmd.extend(
                [
                    "-q:v",
                    str(quality),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2",  # Output format
                    "-vcodec",
                    "mjpeg",  # Ensure JPEG codec
                    "pipe:1",  # Write to stdout
                ]
            )

            # Run ffmpeg with pipes
            process = subprocess.run(
                cmd,
                input=image_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.returncode != 0:
                # If ffmpeg fails, return original or best result so far
                return best_result

            compressed_bytes = process.stdout
            size_kb = len(compressed_bytes) / 1024

            # Keep track of best result
            size_diff = abs(size_kb - target_size_kb)
            if size_kb <= target_size_kb and size_diff < best_size_diff:
                best_result = compressed_bytes
                best_size_diff = size_diff

            if target_size_kb * 0.9 <= size_kb <= target_size_kb:
                # File size is within target range
                return compressed_bytes
            elif size_kb > target_size_kb and quality < 31:
                # Too large → reduce quality
                quality += 2
            elif size_kb < target_size_kb * 0.7 and quality > 2:
                # Too small → improve quality
                quality -= 2
            else:
                # Either quality limits reached or already "good enough"
                break
        return best_result

    @staticmethod
    def save_png_as_rgb(
        image_source: Union[BytesIO, bytes], dest_path: str, crop_box: tuple = None
    ) -> None:
        """
        Save a PNG image as RGB, converting if necessary.

        Args:
            image_source: Either BytesIO object or raw bytes of the image
            dest_path: Destination file path
            crop_box: Optional tuple (left, top, right, bottom) for cropping
        """
        if isinstance(image_source, bytes):
            image_source = BytesIO(image_source)

        with Image.open(image_source) as img:
            if crop_box:
                img = img.crop(crop_box)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest_path, format="PNG")


class PlaywrightHandler:
    @staticmethod
    async def _fetch_content(url: str, scroll_pause_time: float = 2.0) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_extra_http_headers(HEADERS)
            await page.goto(url, wait_until="networkidle")

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
    def fetch_page_content(url: str, method: str = "scroll_to_bottom", **kwargs) -> str:
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
            return asyncio.run(PlaywrightHandler._fetch_content(url, **kwargs))
        elif method == "element_check":
            return asyncio.run(
                PlaywrightHandler._fetch_content_with_element_check(url, **kwargs)
            )
        elif method == "incremental":
            return asyncio.run(
                PlaywrightHandler._fetch_content_incremental(url, **kwargs)
            )
        else:
            raise ValueError(f"Unknown method: {method}")


class AlbumScraper:
    def __init__(self, album_url: str, base_output_dir: str, logger: logging.Logger):
        self.album_url = album_url
        self.album_id = album_url.rstrip("/").split("/")[-1]
        self.logger = logger

        # create subdirectory with name `album_id`
        self.subdir_path = os.path.join(base_output_dir, self.album_id)
        os.makedirs(self.subdir_path, exist_ok=True)
        self.logger.info(f"Created/Using directory: {self.subdir_path}")

    def get_files_from_album(self, raw_html):
        """
        Looks through each item block in the raw html, and extracts the preview png, file name, and file_href
        """
        tree = html.fromstring(raw_html)

        results = []
        for block in tree.xpath("//div[contains(@class,'theItem')]"):
            png_link = block.xpath(
                ".//img[contains(@class,'grid-images_box-img')]/@src"
            )
            file_name = block.xpath(".//p[contains(@class,'theName')]/text()")
            file_href = block.xpath(".//a[starts-with(@href, '/f/')]/@href")

            # make sure values exist before accessing
            png_link = png_link[0] if png_link else None
            file_name = file_name[0].strip() if file_name else None
            file_href = file_href[0] if file_href else None

            if all([png_link, file_name, file_href]):
                results.append(
                    {
                        "preview_png_link": png_link,
                        "file_name": file_name,
                        "file_href": file_href,
                    }
                )

        return results

    def download_preview_png(self, png_url):
        """returns the number of images successfully downloaded"""
        filename = os.path.basename(png_url)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{SUFFIX_PREVIEW}{ext}"
        dest_path = os.path.join(self.subdir_path, new_filename)

        if os.path.exists(dest_path):  # Skip if file already exists
            return 1

        try:
            resp = requests.get(
                png_url,
                headers={**HEADERS, "Referer": self.album_url},
                stream=True,
                timeout=20,
            )
            resp.raise_for_status()

            ImageProcessor.save_png_as_rgb(resp.content, dest_path)
            return 1
        except Exception as e:
            # self.logger.error(f"Failed to download preview {png_url}: {e}") # too verbose
            return 0

    def _check_image_exists(self, base_name: str, original_ext: str) -> bool:
        """Check if either compressed or original image variant exists."""
        compressed_path = os.path.join(
            self.subdir_path, f"{base_name}{SUFFIX_COMPRESSED}.jpg"
        )
        original_path = os.path.join(
            self.subdir_path, f"{base_name}{SUFFIX_ORIGINAL}{original_ext}"
        )

        return os.path.exists(compressed_path) or os.path.exists(original_path)

    def _check_video_exists(self, base_name: str, ext: str) -> bool:
        """Check if both video screenshot files exist."""
        tr_path = os.path.join(self.subdir_path, f"{base_name}{SUFFIX_VIDEO_TR}{ext}")
        bl_path = os.path.join(self.subdir_path, f"{base_name}{SUFFIX_VIDEO_BL}{ext}")

        return os.path.exists(tr_path) and os.path.exists(bl_path)

    def download_file(self, file_url: str, file_name: str = None):
        """
        given a link to a file `file_url`, grab the file in the page
        """

        def process_video_preview(resp, filename):
            image_data = BytesIO(resp.content)  # we assume we received a PNG
            with Image.open(image_data) as img:
                width, height = img.size

                # Process quarter beneath top-right quarter
                crop_box_tr = (width * 3 // 4, height // 4, width, height // 2)
                base_name, ext = os.path.splitext(filename)
                tr_filename = f"{base_name}{SUFFIX_VIDEO_TR}{ext}"
                tr_dest_path = os.path.join(self.subdir_path, tr_filename)
                ImageProcessor.save_png_as_rgb(
                    resp.content, tr_dest_path, crop_box=crop_box_tr
                )

                # Process bottom-left quarter
                crop_box_bl = (0, height * 3 // 4, width // 4, height)
                bl_filename = f"{base_name}{SUFFIX_VIDEO_BL}{ext}"
                bl_dest_path = os.path.join(self.subdir_path, bl_filename)
                ImageProcessor.save_png_as_rgb(
                    resp.content, bl_dest_path, crop_box=crop_box_bl
                )
            return

        def process_image(resp, filename, src_url):
            content_bytes = resp.content
            size_kb = len(content_bytes) / 1024

            base_name, ext = os.path.splitext(filename)

            if size_kb > MAX_SIZE_KB:
                compressed_bytes = ImageProcessor.compress_image_in_memory(
                    content_bytes, MAX_SIZE_KB
                )
                compressed_size_kb = len(compressed_bytes) / 1024

                # Save compressed image with .jpg extension
                file_name = f"{base_name}{SUFFIX_COMPRESSED}"
                dest_path = os.path.join(self.subdir_path, f"{file_name}.jpg")
                with open(dest_path, "wb") as f:
                    f.write(compressed_bytes)
            else:
                # Direct save for images under size limit or non-image files
                file_name = f"{base_name}{SUFFIX_ORIGINAL}{ext}"
                dest_path = os.path.join(self.subdir_path, file_name)
                if src_url.endswith(".png"):
                    ImageProcessor.save_png_as_rgb(content_bytes, dest_path)
                else:
                    # Direct save for non-PNG images
                    with open(dest_path, "wb") as f:
                        f.write(content_bytes)
            return

        try:
            raw_html = PlaywrightHandler.fetch_page_content(file_url)
            tree = html.fromstring(raw_html)

            ### FIRST PROCESS AS IF IT'S A VIDEO
            containers = tree.xpath(
                '//div[@class="rounded-lg"]'
            )  # Narrow scope: find the first <div class="rounded-lg">
            if containers:
                container = containers[0]
                scripts = container.xpath(
                    ".//script/text()"
                )  # Extract all <script> inside that container
                video_cover_url = None
                for script in scripts:
                    match = re.search(
                        r'var\s+videoCoverUrl\s*=\s*"([^"]+_grid\.png)"', script
                    )
                    if match:
                        video_cover_url = match.group(1).replace(
                            "\\/", "/"
                        )  # unescape "\/"
                        break
                if video_cover_url:  # IF VIDEO PATTERN MATCHES, GRAB THE PNG
                    filename = os.path.basename(video_cover_url)
                    base_name, ext = os.path.splitext(filename)

                    # Check if both video files already exist
                    if self._check_video_exists(base_name, ext):
                        # self.logger.info(f"Video preview files already exist for: {filename}")
                        return "vid"

                    resp = requests.get(
                        video_cover_url,
                        headers={**HEADERS, "Referer": file_url},
                        stream=True,
                        timeout=20,
                    )
                    resp.raise_for_status()
                    process_video_preview(resp, filename)
                    return "vid"

            ### OTHERWISE PROCESS IT AS AN IMAGE
            img_sources = tree.xpath('//main//img[contains(@src, "reddit")]/@src')
            if img_sources:
                img_url = img_sources[0]  # grab first match

                filename = os.path.basename(img_url)
                base_name, ext = os.path.splitext(filename)

                # Check if either image variant already exists
                if self._check_image_exists(base_name, ext):
                    # self.logger.info(f"Image file already exists for: {filename}")
                    return "pic"

                resp = requests.get(
                    img_url,
                    headers={**HEADERS, "Referer": file_url},
                    stream=True,
                    timeout=20,
                )
                resp.raise_for_status()
                process_image(resp, filename, img_url)
                return "pic"

            return None  ### else we return nothing
        except Exception as e:
            self.logger.error(f"Failed to download file from {file_url}: {e}")
            return None

    def process_album(self, album_url: str, shutdown_event: mp.Event):
        break_flag = False  # bool: did we get terminated?

        album_html = PlaywrightHandler.fetch_page_content(album_url + "?advanced=1")
        file_list = self.get_files_from_album(album_html)
        self.logger.info(f"Found {len(file_list)} files in HTML from {self.album_id}")

        preview_count = 0
        pic_count = 0
        vid_count = 0
        for file in file_list:
            # time.sleep(1) # to prevent overloading server

            # Check if we should terminate
            if shutdown_event.is_set():
                self.logger.info(f"Shutdown requested, stopping album {self.album_id}")
                break_flag = True
                break

            ### first grab the preview
            preview_count += self.download_preview_png(file["preview_png_link"])

            ### then grab the file
            #
            # assemble file url from href
            parsed = urlparse(album_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            file_url = urljoin(base, file["file_href"])
            file_result = self.download_file(file_url)

            if file_result == "pic":
                pic_count += 1
            elif file_result == "vid":
                vid_count += 1

        # if a file is in an unsupported format, then it should not count towards num_total
        file_list = [
            f
            for f in file_list
            if os.path.splitext(f["file_name"].lower())[1]
            not in UNDESIREABLE_EXTENSIONS
        ]

        return AlbumResult(
            album_url=self.album_url,
            album_id=self.album_id,
            num_previews=preview_count,
            num_files=pic_count + vid_count,
            num_total=len(file_list),
            incomplete=break_flag,
        )


def setup_album_logger(album_id: str, log_dir: Path) -> logging.Logger:
    """Set up a logger for each album that writes to its own subdirectory."""
    album_dir = log_dir / album_id
    album_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"album_{album_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers

    # Create file handler
    log_file = album_dir / "scraper.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def worker_function(
    worker_id: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    base_output_dir: str,
):
    """Worker function that processes album jobs from the queue."""
    # Set up base logger for worker
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    worker_logger.addHandler(handler)

    worker_logger.info(f"Worker {worker_id} started")

    try:
        while not shutdown_event.is_set():
            try:
                # Try to get a job with timeout to periodically check shutdown
                job = job_queue.get(timeout=1.0)

                if job is None:  # Poison pill
                    worker_logger.info(f"Worker {worker_id} received poison pill")
                    break

                album_url = job.get("album_url")
                worker_logger.info(f"Worker {worker_id} processing album: {album_url}")

                # Extract album ID from URL
                album_id = album_url.rstrip("/").split("/")[-1]

                # Set up album-specific logger
                album_logger = setup_album_logger(album_id, Path(base_output_dir))

                # Create scraper instance and process album
                scraper = AlbumScraper(album_url, base_output_dir, album_logger)
                result = scraper.process_album(album_url, shutdown_event)

                # Put result in result queue
                result_queue.put(result)
                worker_logger.info(f"Worker {worker_id} completed album: {album_url}")

            except queue.Empty:
                # No job available, continue to check shutdown
                continue
            except Exception as e:
                worker_logger.error(
                    f"Worker {worker_id} error processing job: {e}", exc_info=True
                )

    except KeyboardInterrupt:
        worker_logger.info(f"Worker {worker_id} interrupted")
    finally:
        worker_logger.info(f"Worker {worker_id} shutting down")


def result_collector(
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    logger: logging.Logger,
    completed_file: Path,
):
    """Periodically collect results from the result queue and log completed albums."""
    accumulated_results = []

    while not shutdown_event.is_set() or not result_queue.empty():
        try:
            # Try to get results with timeout
            result = result_queue.get(timeout=30)
            accumulated_results.append(result)
            logger.info(f"Collected result for album: {result.album_id}")

            # Process accumulated results periodically (batches of 16)
            if len(accumulated_results) >= 16:
                process_accumulated_results(accumulated_results, logger, completed_file)
                accumulated_results.clear()

        except queue.Empty:
            # Process any remaining results
            if accumulated_results:
                process_accumulated_results(accumulated_results, logger, completed_file)
                accumulated_results.clear()

    # Process any final results
    if accumulated_results:
        process_accumulated_results(accumulated_results, logger, completed_file)


def process_accumulated_results(
    results: list[AlbumResult], logger: logging.Logger, completed_file: Path
):
    """Process a batch of accumulated results and log to completed_links.txt."""
    logger.info(f"Processing {len(results)} accumulated results")

    with open(completed_file, "a") as f:
        for result in results:
            if result.incomplete == False and (
                result.num_total == 0 or result.num_files / result.num_total > 0.9
            ):
                # Log to completed_links.txt
                f.write(f"{result.album_url}\n")

                # Log summary
                status = "INCOMPLETE" if result.incomplete else "COMPLETE"
                logger.info(
                    f"Album {result.album_id}: {status} - "
                    f"Previews: {result.num_previews}/{result.num_total}, "
                    f"Files: {result.num_files}/{result.num_total}"
                )
            else:
                logger.info(
                    f"REJECTED ALBUM, CONSIDERED INCOMPLETE: "
                    f"Album {result.album_id} "
                    f"Previews: {result.num_previews}/{result.num_total}, "
                    f"Files: {result.num_files}/{result.num_total}"
                )


def signal_handler(signum, frame, shutdown_event: mp.Event, logger: logging.Logger):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


def load_album_links(file_path: str) -> list[str]:
    """Load album links from a file."""
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def main():
    """Main function that manages the job queue and workers."""
    # Set up logging for main process
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    # Configuration
    base_output_dir = "BAD"
    os.makedirs(base_output_dir, exist_ok=True)

    completed_file = Path("completed_links.txt")
    album_links_file = "bad_links.txt"

    # Load album links
    album_links = load_album_links(album_links_file)
    if not album_links:
        logger.error(f"No album links found in {album_links_file}")
        return

    # Load already completed albums
    completed_albums = set()
    if completed_file.exists():
        with open(completed_file, "r") as f:
            completed_albums = set(line.strip() for line in f if line.strip())

    # Filter out already completed albums
    pending_albums = [url for url in album_links if url not in completed_albums]
    logger.info(f"Found {len(pending_albums)} pending albums to process")

    if not pending_albums:
        logger.info("All albums already processed!")
        return

    # Number of worker processes
    n_workers = 15  # min(mp.cpu_count() - 1 or 1, 8)  # Cap at 8 workers
    logger.info(f"Starting with {n_workers} workers")

    # Create queues and shutdown event
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    shutdown_event = mp.Event()

    # Set up signal handlers for graceful shutdown
    def handler(signum, frame):
        signal_handler(signum, frame, shutdown_event, logger)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # Start worker processes
    workers = []
    for i in range(n_workers):
        worker = mp.Process(
            target=worker_function,
            args=(i, job_queue, result_queue, shutdown_event, base_output_dir),
        )
        worker.start()
        workers.append(worker)
        logger.info(f"Started worker {i}")

    # Start result collector in a separate thread
    collector_thread = threading.Thread(
        target=result_collector,
        args=(result_queue, shutdown_event, logger, completed_file),
    )
    collector_thread.start()

    try:
        # Enqueue album jobs
        for album_url in pending_albums:
            if shutdown_event.is_set():
                break

            job = {"album_url": album_url, "timestamp": time.time()}
            job_queue.put(job)
            logger.info(f"Enqueued album: {album_url}")

        # Wait for all jobs to be processed
        logger.info("All albums enqueued, waiting for completion...")

        # Send poison pills to workers
        for _ in range(n_workers):
            job_queue.put(None)

        # Wait for workers to finish
        for worker in workers:
            worker.join()
            if worker.is_alive():
                logger.warning(f"Worker {worker.pid} did not shut down gracefully")
                worker.terminate()

        # Signal collector thread to stop
        shutdown_event.set()
        collector_thread.join(timeout=10)

    except KeyboardInterrupt:
        logger.info("Main process interrupted")
        shutdown_event.set()
    finally:
        # Ensure all processes are terminated
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()

        logger.info("All workers stopped")
        logger.info("Main process exiting")


if __name__ == "__main__":
    # Required for Windows
    mp.freeze_support()
    main()
