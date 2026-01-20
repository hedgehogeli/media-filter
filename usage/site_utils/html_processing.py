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
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
from lxml import html
from PIL import Image


class site_html_processor():
    def __init__():
        pass

    @staticmethod
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