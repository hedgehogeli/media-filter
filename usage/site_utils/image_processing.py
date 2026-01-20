
import subprocess
from typing import Optional


MAX_SIZE_KB = 500
MAX_DIMENSION = 1920

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
    def compress(
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