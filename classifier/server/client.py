import logging
import socket
import time
from abc import ABC, abstractmethod
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple

from .common_util import (
    BatchResponse,
    ClientBatch,
    ImageData,
    ProtocolDecoder,
    ProtocolEncoder,
    QueueFullResponse,
)


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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import os
    
    client = MLBatchClient(server_host="127.0.0.1", server_port=5000) # 192.168.56.1
    
    try:
        client.connect()
        
        # Load test images
        images = []

        # Load actual images
        for img_path in ["image1.jpg", "image2.jpg"]:  # Limit to 5 images
            with open(img_path, 'rb') as f: # f.read() returns "bytes"
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
                print(f"    Scores: [{score.score1:.3f}, {score.score2:.3f}, {score.score3:.3f}]")
        
        # Optional: Keep connection alive for more batches
        # print("\nConnection remains open for additional batches...")
        # time.sleep(2)  # Or wait for user input
        
    except Exception as e:
        print(f"Client error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()