# self contained client side code for copy pastig to actual client

import json
import logging
import socket
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple, Union

#################################### COMMON_UTIL snippets below

@dataclass
class ImageScore:
    """ML model output for a single image"""

    image_name: str
    score1: float
    score2: float
    score3: float
    error: Optional[str] = None


@dataclass
class ImageData:
    """Raw image data from client"""

    filename: str
    data: bytes

    def validate(self) -> None:
        """Validate image data constraints"""
        if len(self.data) > 1024 * 1024:  # 1MB limit
            raise ValueError(f"Image {self.filename} exceeds 1MB limit")


@dataclass
class QueueFullResponse:
    """Response when server queue is full"""

    status: str = "QUEUE_FULL"
    batch_id: str = ""


@dataclass
class BatchResponse:
    """Response for a client batch"""

    batch_id: str
    processing_time: float
    scores: List[ImageScore]


@dataclass
class ClientBatch:
    """Batch of images from client (1-10 images)"""

    batch_id: str
    images: List[ImageData]
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None

    def validate(self) -> None:
        """Validate batch constraints"""
        if not (1 <= len(self.images) <= 10):
            raise ValueError(f"Batch must contain 1-10 images, got {len(self.images)}")
        for img in self.images:
            img.validate()


@dataclass
class ImageData:
    """Raw image data from client"""

    filename: str
    data: bytes

    def validate(self) -> None:
        """Validate image data constraints"""
        if len(self.data) > 1024 * 1024:  # 1MB limit
            raise ValueError(f"Image {self.filename} exceeds 1MB limit")


class ProtocolDecoder:
    """Decodes data from network transmission"""

    @staticmethod
    def receive_exact(sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exact number of bytes from socket"""
        data = bytearray()
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed while receiving data")
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    def decode_batch_request(sock: socket.socket) -> ClientBatch:
        """Decode client batch from network socket"""
        # Read number of images
        num_images_data = ProtocolDecoder.receive_exact(sock, 4)
        num_images = struct.unpack("!I", num_images_data)[0]

        # Read batch ID
        batch_id_len_data = ProtocolDecoder.receive_exact(sock, 4)
        batch_id_len = struct.unpack("!I", batch_id_len_data)[0]
        batch_id = ProtocolDecoder.receive_exact(sock, batch_id_len).decode("utf-8")

        # Read images
        images = []
        for _ in range(num_images):
            # Filename
            filename_len_data = ProtocolDecoder.receive_exact(sock, 4)
            filename_len = struct.unpack("!I", filename_len_data)[0]
            filename = ProtocolDecoder.receive_exact(sock, filename_len).decode("utf-8")

            # Image data
            image_len_data = ProtocolDecoder.receive_exact(sock, 4)
            image_len = struct.unpack("!I", image_len_data)[0]
            image_data = ProtocolDecoder.receive_exact(sock, image_len)

            images.append(ImageData(filename=filename, data=image_data))

        return ClientBatch(batch_id=batch_id, images=images)

    @staticmethod
    def decode_batch_response(
        sock: socket.socket,
    ) -> Union[BatchResponse, QueueFullResponse]:
        """Decode server response from network socket"""
        # Read batch ID
        batch_id_len_data = ProtocolDecoder.receive_exact(sock, 4)
        batch_id_len = struct.unpack("!I", batch_id_len_data)[0]
        batch_id = ProtocolDecoder.receive_exact(sock, batch_id_len).decode("utf-8")

        # Read JSON
        json_len_data = ProtocolDecoder.receive_exact(sock, 4)
        json_len = struct.unpack("!I", json_len_data)[0]
        json_data = ProtocolDecoder.receive_exact(sock, json_len).decode("utf-8")

        response_dict = json.loads(json_data)

        if response_dict.get("status") == "QUEUE_FULL":
            return QueueFullResponse(batch_id=batch_id)
        else:
            scores = [
                ImageScore(
                    image_name=s["image_name"],
                    score1=s["score1"],
                    score2=s["score2"],
                    score3=s["score3"],
                    error=s.get("error"),
                )
                for s in response_dict["scores"]
            ]
            return BatchResponse(
                batch_id=response_dict["batch_id"],
                processing_time=response_dict["processing_time"],
                scores=scores,
            )


class ProtocolEncoder:
    """Encodes data for network transmission"""

    @staticmethod
    def encode_batch_request(batch: ClientBatch) -> bytes:
        """Encode client batch for transmission"""
        data = bytearray()

        # Number of images
        data.extend(struct.pack("!I", len(batch.images)))

        # Batch ID
        batch_id_bytes = batch.batch_id.encode("utf-8")
        data.extend(struct.pack("!I", len(batch_id_bytes)))
        data.extend(batch_id_bytes)

        # Each image
        for img in batch.images:
            # Filename
            filename_bytes = img.filename.encode("utf-8")
            data.extend(struct.pack("!I", len(filename_bytes)))
            data.extend(filename_bytes)

            # Image data
            data.extend(struct.pack("!I", len(img.data)))
            data.extend(img.data)

        return bytes(data)

    @staticmethod
    def encode_batch_response(
        response: Union[BatchResponse, QueueFullResponse],
    ) -> bytes:
        """Encode server response for transmission"""
        if isinstance(response, QueueFullResponse):
            response_dict = {"status": response.status, "batch_id": response.batch_id}
        else:
            response_dict = {
                "batch_id": response.batch_id,
                "processing_time": response.processing_time,
                "scores": [
                    {
                        "image_name": score.image_name,
                        "score1": score.score1,
                        "score2": score.score2,
                        "score3": score.score3,
                        "error": score.error,
                    }
                    for score in response.scores
                ],
            }

        json_bytes = json.dumps(response_dict).encode("utf-8")

        data = bytearray()
        # Batch ID length and data
        batch_id_bytes = response_dict["batch_id"].encode("utf-8")
        data.extend(struct.pack("!I", len(batch_id_bytes)))
        data.extend(batch_id_bytes)

        # JSON length and data
        data.extend(struct.pack("!I", len(json_bytes)))
        data.extend(json_bytes)

        return bytes(data)


@dataclass
class QueueFullResponse:
    """Response when server queue is full"""

    status: str = "QUEUE_FULL"
    batch_id: str = ""


############################################################################### CLIENT BELOW


class ResponseCallback(ABC):
    """Abstract base for response callbacks"""

    @abstractmethod
    def on_response(self, response: BatchResponse) -> None:
        """Handle successful response"""
        pass

    @abstractmethod
    def on_error(self, batch_id: str, error: Exception) -> None:
        """Handle error response"""
        pass


class SyncResponseCallback(ResponseCallback):
    """Synchronous response callback"""

    def __init__(self):
        self.response: Optional[BatchResponse] = None
        self.error: Optional[Exception] = None
        self.event = Event()

    def on_response(self, response: BatchResponse) -> None:
        self.response = response
        self.event.set()

    def on_error(self, batch_id: str, error: Exception) -> None:
        self.error = error
        self.event.set()

    def wait(self, timeout: float) -> BatchResponse:
        if not self.event.wait(timeout):
            raise TimeoutError("Response timeout")
        if self.error:
            raise self.error
        return self.response


class MLBatchClient:
    """Client for submitting image batches"""

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 5000,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 5,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self.socket: Optional[socket.socket] = None
        self.socket_lock = Lock()
        self.connected = False
        self.batch_counter = 0
        self.pending_responses: Dict[str, ResponseCallback] = {}
        self.response_lock = Lock()
        self.response_thread: Optional[Thread] = None
        self.logger = logging.getLogger(__name__)
        self.running = True

    def connect(self) -> None:
        """Establish connection to server"""
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.server_host, self.server_port))
                self.connected = True
                self.logger.info(
                    f"Connected to server at {self.server_host}:{self.server_port}"
                )

                # Start response thread
                self.response_thread = Thread(
                    target=self._receive_responses, name="ClientReceiver"
                )
                self.response_thread.daemon = True
                self.response_thread.start()

                return
            except Exception as e:
                attempts += 1
                self.logger.error(f"Connection attempt {attempts} failed: {e}")
                if attempts < self.max_reconnect_attempts:
                    time.sleep(self.reconnect_delay)

        raise ConnectionError(f"Failed to connect after {attempts} attempts")

    def disconnect(self) -> None:
        """Close connection to server"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=2.0)

    def submit_batch(
        self,
        images: List[Tuple[str, bytes]],
        callback: Optional[ResponseCallback] = None,
    ) -> str:
        """Submit batch of images for processing"""
        if not self.connected:
            raise ConnectionError("Not connected to server")

        # Create batch
        batch_id = self._generate_batch_id()
        image_data = [ImageData(filename=name, data=data) for name, data in images]
        batch = ClientBatch(batch_id=batch_id, images=image_data)
        batch.validate()

        # Register callback
        if callback:
            with self.response_lock:
                self.pending_responses[batch_id] = callback

        # Send batch
        try:
            self._send_batch(batch)
            return batch_id
        except Exception as e:
            if callback:
                with self.response_lock:
                    if batch_id in self.pending_responses:
                        del self.pending_responses[batch_id]
            raise

    def submit_batch_sync(
        self, images: List[Tuple[str, bytes]], timeout: float = 30.0
    ) -> BatchResponse:
        """Submit batch and wait for response"""
        callback = SyncResponseCallback()
        self.submit_batch(images, callback)
        return callback.wait(timeout)

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        with self.socket_lock:
            self.batch_counter += 1
            return f"batch_{self.batch_counter:06d}"

    def _send_batch(self, batch: ClientBatch) -> None:
        """Send batch over socket with thread safety"""
        data = ProtocolEncoder.encode_batch_request(batch)
        with self.socket_lock:
            self.socket.sendall(data)

    def _receive_responses(self) -> None:
        """Background thread receiving responses"""
        while self.running and self.connected:
            try:
                response = ProtocolDecoder.decode_batch_response(self.socket)

                if isinstance(response, QueueFullResponse):
                    self._handle_queue_full(response.batch_id)
                else:
                    # Find and call callback
                    with self.response_lock:
                        callback = self.pending_responses.pop(response.batch_id, None)

                    if callback:
                        try:
                            callback.on_response(response)
                        except Exception as e:
                            self.logger.error(f"Callback error: {e}")

            except ConnectionError as e:
                if self.running:
                    # Only log as error if we're not shutting down
                    self.logger.error("Connection lost")
                    self.connected = False
                break
            except Exception as e:
                if self.running:
                    self.logger.error(f"Error receiving response: {e}", exc_info=True)

    def _handle_queue_full(self, batch_id: str) -> None:
        """Handle queue full response with exponential backoff"""
        self.logger.warning(f"Server queue full for batch {batch_id}")
        # In a real implementation, we would retry with backoff
        with self.response_lock:
            callback = self.pending_responses.pop(batch_id, None)

        if callback:
            callback.on_error(batch_id, Exception("Server queue full"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import os

    client = MLBatchClient(server_host="127.0.0.1", server_port=5000)  # 192.168.56.1

    try:
        client.connect()

        # Load test images
        images = []

        # Load actual images
        for img_path in ["image1.jpg", "image2.jpg"]:  # Limit to 5 images
            with open(img_path, "rb") as f:  # f.read() returns "bytes"
                images.append((os.path.basename(img_path), f.read()))
            print(f"Loaded {img_path}")

        # Submit batch
        print(f"Submitting batch of {len(images)} images...")
        response = client.submit_batch_sync(images, timeout=30.0)

        # Print results
        print(f"\nBatch ID: {response.batch_id}")
        print(f"Processing time: {response.processing_time:.3f}s")
        print("\nResults:")
        for score in response.scores:
            print(f"  {score.image_name}:")
            if score.error:
                print(f"    Error: {score.error}")
            else:
                print(
                    f"    Scores: [{score.score1:.3f}, {score.score2:.3f}, {score.score3:.3f}]"
                )

        # Optional: Keep connection alive for more batches
        # print("\nConnection remains open for additional batches...")
        # time.sleep(2)  # Or wait for user input

    except Exception as e:
        print(f"Client error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        client.disconnect()
