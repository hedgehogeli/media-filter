import json
import struct
import socket
import logging
import time
import signal
import sys
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from threading import Thread, Lock, Event
from queue import Queue, Empty, Full
from datetime import datetime
from abc import ABC, abstractmethod
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from model import load_for_inference, to_tensor, val_transform


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ==================== Data Contracts ====================

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
class ProcessedImage:
    """Image after transform processing"""
    filename: str
    tensor: Optional[torch.Tensor] = None  # CPU tensor of shape [3, 224, 224]
    error: Optional[str] = None

@dataclass
class ImageScore:
    """ML model output for a single image"""
    image_name: str
    score1: float
    score2: float
    score3: float
    error: Optional[str] = None

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
class ModelBatch:
    """Assembled batch for model inference (up to 40 images)"""
    batch_ids: List[str]  # Client batch IDs included
    images: List[ProcessedImage]
    client_batches: Dict[str, List[ProcessedImage]]  # Maintain client batch grouping
    assembly_time: float = 0.0
    
    def get_tensor_batch(self) -> Tuple[torch.Tensor, List[int]]:
        """Stack valid tensors for model input, return tensor and valid indices"""
        valid_tensors = []
        valid_indices = []
        for i, img in enumerate(self.images):
            if img.error is None and img.tensor is not None:
                valid_tensors.append(img.tensor)
                valid_indices.append(i)
        
        if not valid_tensors:
            raise ValueError("No valid images in batch")
        return torch.stack(valid_tensors), valid_indices

@dataclass
class BatchResponse:
    """Response for a client batch"""
    batch_id: str
    processing_time: float
    scores: List[ImageScore]

@dataclass
class QueueFullResponse:
    """Response when server queue is full"""
    status: str = "QUEUE_FULL"
    batch_id: str = ""

# ==================== Network Protocol ====================

class ProtocolEncoder:
    """Encodes data for network transmission"""
    
    @staticmethod
    def encode_batch_request(batch: ClientBatch) -> bytes:
        """Encode client batch for transmission"""
        data = bytearray()
        
        # Number of images
        data.extend(struct.pack('!I', len(batch.images)))
        
        # Batch ID
        batch_id_bytes = batch.batch_id.encode('utf-8')
        data.extend(struct.pack('!I', len(batch_id_bytes)))
        data.extend(batch_id_bytes)
        
        # Each image
        for img in batch.images:
            # Filename
            filename_bytes = img.filename.encode('utf-8')
            data.extend(struct.pack('!I', len(filename_bytes)))
            data.extend(filename_bytes)
            
            # Image data
            data.extend(struct.pack('!I', len(img.data)))
            data.extend(img.data)
        
        return bytes(data)
    
    @staticmethod
    def encode_batch_response(response: Union[BatchResponse, QueueFullResponse]) -> bytes:
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
                        "error": score.error
                    }
                    for score in response.scores
                ]
            }
        
        json_bytes = json.dumps(response_dict).encode('utf-8')
        
        data = bytearray()
        # Batch ID length and data
        batch_id_bytes = response_dict["batch_id"].encode('utf-8')
        data.extend(struct.pack('!I', len(batch_id_bytes)))
        data.extend(batch_id_bytes)
        
        # JSON length and data
        data.extend(struct.pack('!I', len(json_bytes)))
        data.extend(json_bytes)
        
        return bytes(data)

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
        num_images = struct.unpack('!I', num_images_data)[0]
        
        # Read batch ID
        batch_id_len_data = ProtocolDecoder.receive_exact(sock, 4)
        batch_id_len = struct.unpack('!I', batch_id_len_data)[0]
        batch_id = ProtocolDecoder.receive_exact(sock, batch_id_len).decode('utf-8')
        
        # Read images
        images = []
        for _ in range(num_images):
            # Filename
            filename_len_data = ProtocolDecoder.receive_exact(sock, 4)
            filename_len = struct.unpack('!I', filename_len_data)[0]
            filename = ProtocolDecoder.receive_exact(sock, filename_len).decode('utf-8')
            
            # Image data
            image_len_data = ProtocolDecoder.receive_exact(sock, 4)
            image_len = struct.unpack('!I', image_len_data)[0]
            image_data = ProtocolDecoder.receive_exact(sock, image_len)
            
            images.append(ImageData(filename=filename, data=image_data))
        
        return ClientBatch(batch_id=batch_id, images=images)
    
    @staticmethod
    def decode_batch_response(sock: socket.socket) -> Union[BatchResponse, QueueFullResponse]:
        """Decode server response from network socket"""
        # Read batch ID
        batch_id_len_data = ProtocolDecoder.receive_exact(sock, 4)
        batch_id_len = struct.unpack('!I', batch_id_len_data)[0]
        batch_id = ProtocolDecoder.receive_exact(sock, batch_id_len).decode('utf-8')
        
        # Read JSON
        json_len_data = ProtocolDecoder.receive_exact(sock, 4)
        json_len = struct.unpack('!I', json_len_data)[0]
        json_data = ProtocolDecoder.receive_exact(sock, json_len).decode('utf-8')
        
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
                    error=s.get("error")
                )
                for s in response_dict["scores"]
            ]
            return BatchResponse(
                batch_id=response_dict["batch_id"],
                processing_time=response_dict["processing_time"],
                scores=scores
            )

# ==================== Server Components ====================

class ConnectionHandler(Thread):
    """Handles individual client connections"""
    
    def __init__(self, 
                 client_socket: socket.socket,
                 client_address: Tuple[str, int],
                 batch_queue: 'BatchQueue',
                 response_manager: 'ResponseManager',
                 logger: logging.Logger):
        super().__init__(name=f"Handler-{client_address}")
        self.client_socket = client_socket
        self.client_address = client_address
        self.batch_queue = batch_queue
        self.response_manager = response_manager
        self.logger = logger
        self.client_id = f"{client_address[0]}:{client_address[1]}"
        self.running = True
        self.daemon = True
    
    def run(self) -> None:
        """Main connection handling loop"""
        self.logger.info(f"Handling connection from {self.client_id}")
        try:
            while self.running:
                try:
                    batch = self._receive_batch()
                    if batch:
                        batch.client_id = self.client_id
                        if not self.batch_queue.put_nowait(batch):
                            # Queue is full
                            response = QueueFullResponse(batch_id=batch.batch_id)
                            self._send_response(response)
                        else:
                            # Register for response
                            self.response_manager.register_handler(batch.batch_id, self)
                    else:
                        # Connection closed by client
                        break
                except ConnectionError:
                    # Client disconnected
                    break
                except Exception as e:
                    self.logger.error(f"Error handling batch: {e}", exc_info=True)
                    # Continue handling connection unless it's a critical error
                    if isinstance(e, (BrokenPipeError, ConnectionResetError)):
                        break
        finally:
            self.response_manager.unregister_client(self.client_id)
            self.client_socket.close()
            self.logger.info(f"Connection closed for {self.client_id}")
    
    def _receive_batch(self) -> Optional[ClientBatch]:
        """Receive and decode a batch from client"""
        try:
            return ProtocolDecoder.decode_batch_request(self.client_socket)
        except ConnectionError:
            return None
    
    def _send_response(self, response: Union[BatchResponse, QueueFullResponse]) -> None:
        """Send response back to client"""
        try:
            data = ProtocolEncoder.encode_batch_response(response)
            self.client_socket.sendall(data)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
            raise

    def stop(self):
        """Stop the handler"""
        self.running = False

class BatchQueue:
    """Thread-safe queue for client batches with size limit"""
    
    def __init__(self, max_size: int = 10):
        self.queue: Queue[ClientBatch] = Queue(maxsize=max_size)
        self.logger = logging.getLogger(__name__)
    
    def put_nowait(self, batch: ClientBatch) -> bool:
        """Try to add batch, return False if full"""
        try:
            self.queue.put_nowait(batch)
            self.logger.info(f"Added batch {batch.batch_id} to queue")
            return True
        except Full:
            self.logger.warning(f"Queue full, rejecting batch {batch.batch_id}")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[ClientBatch]:
        """Get batch from queue with optional timeout"""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

def transform_image(image: Image.Image) -> torch.Tensor:
    """Transform image using the provided transforms, return CPU tensor"""
    # Convert PIL image to tensor and move to GPU if available
    image_tensor = to_tensor(image)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.to("cuda")
        # Apply transform on GPU
        transformed = val_transform(image_tensor)
        # Move back to CPU for storage
        return transformed.to("cpu")
    else:
        # CPU-only path
        return val_transform(image_tensor)

class TransformWorker(Thread):
    """Worker thread for image transformation"""
    
    def __init__(self,
                 worker_id: int,
                 input_queue: Queue[ClientBatch],
                 output_queue: Queue[Tuple[ClientBatch, List[ProcessedImage]]],
                 logger: logging.Logger,
                 shutdown_event: Event):
        super().__init__(name=f"TransformWorker-{worker_id}")
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logger
        self.shutdown_event = shutdown_event
        self.daemon = True
    
    def run(self) -> None:
        """Main worker loop"""
        self.logger.info(f"Transform worker {self.worker_id} started")
        while not self.shutdown_event.is_set():
            try:
                batch = self.input_queue.get(timeout=0.1)
                if batch is None:
                    continue
                
                start_time = time.time()
                processed = self._process_batch(batch)
                transform_time = time.time() - start_time
                
                self.logger.info(f"Worker {self.worker_id} processed batch {batch.batch_id} "
                               f"({len(batch.images)} images) in {transform_time:.3f}s")
                
                self.output_queue.put((batch, processed))
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Transform worker {self.worker_id} error: {e}", exc_info=True)
    
    def _process_batch(self, batch: ClientBatch) -> List[ProcessedImage]:
        """Transform all images in a client batch"""
        processed = []
        for img_data in batch.images:
            try:
                # Open image from bytes
                image = Image.open(BytesIO(img_data.data))
                # Apply transform
                tensor = transform_image(image)
                processed.append(ProcessedImage(filename=img_data.filename, tensor=tensor))
            except Exception as e:
                self.logger.error(f"Failed to process {img_data.filename}: {e}")
                processed.append(ProcessedImage(
                    filename=img_data.filename,
                    error=f"Transform failed: {str(e)}"
                ))
        return processed

class BatchAssembler(Thread):
    """Assembles client batches into model batches"""
    
    def __init__(self,
                 input_queue: Queue[Tuple[ClientBatch, List[ProcessedImage]]],
                 output_queue: Queue[ModelBatch],
                 logger: logging.Logger,
                 shutdown_event: Event,
                 max_batch_size: int = 40,
                 timeout_seconds: float = 2.0):
        super().__init__(name="BatchAssembler")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.max_batch_size = max_batch_size
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.shutdown_event = shutdown_event
        self.current_assembly: List[Tuple[ClientBatch, List[ProcessedImage]]] = []
        self.assembly_start_time: Optional[float] = None
        self.daemon = True
    
    def run(self) -> None:
        """Main assembly loop"""
        self.logger.info("Batch assembler started")
        while not self.shutdown_event.is_set():
            try:
                # Check timeout
                if self.assembly_start_time and self.current_assembly:
                    if time.time() - self.assembly_start_time > self.timeout_seconds:
                        self._flush_assembly()
                
                # Try to get new batch
                try:
                    client_batch, processed_images = self.input_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Check if adding this batch would exceed limit
                current_size = sum(len(cb[1]) for cb in self.current_assembly)
                incoming_size = len(processed_images)
                
                if self._should_process_current_assembly(current_size, incoming_size):
                    self._flush_assembly()
                
                # Add to current assembly
                self.current_assembly.append((client_batch, processed_images))
                if self.assembly_start_time is None:
                    self.assembly_start_time = time.time()
                
                # Check if we've reached max size
                new_size = sum(len(cb[1]) for cb in self.current_assembly)
                if new_size >= self.max_batch_size:
                    self._flush_assembly()
                    
            except Exception as e:
                self.logger.error(f"Batch assembler error: {e}", exc_info=True)
        
        # Flush any remaining batches on shutdown
        if self.current_assembly:
            self._flush_assembly()
    
    def _should_process_current_assembly(self, current_size: int, incoming_size: int) -> bool:
        """Determine if current assembly should be processed"""
        return current_size > 0 and current_size + incoming_size > self.max_batch_size
    
    def _flush_assembly(self) -> None:
        """Create and queue model batch from current assembly"""
        if not self.current_assembly:
            return
        
        model_batch = self._create_model_batch()
        if model_batch:
            self.output_queue.put(model_batch)
            self.logger.info(f"Assembled model batch with {len(model_batch.images)} images "
                           f"from {len(model_batch.batch_ids)} client batches")
        
        self.current_assembly = []
        self.assembly_start_time = None
    
    def _create_model_batch(self) -> Optional[ModelBatch]:
        """Create model batch from current assembly"""
        if not self.current_assembly:
            return None
        
        batch_ids = []
        all_images = []
        client_batches = {}
        
        for client_batch, processed_images in self.current_assembly:
            batch_ids.append(client_batch.batch_id)
            all_images.extend(processed_images)
            client_batches[client_batch.batch_id] = processed_images
        
        return ModelBatch(
            batch_ids=batch_ids,
            images=all_images,
            client_batches=client_batches,
            assembly_time=time.time() - (self.assembly_start_time or time.time())
        )

class ModelInferenceWorker(Thread):
    """Performs ML model inference on batches"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 input_queue: Queue[ModelBatch],
                 output_queue: Queue[Dict[str, BatchResponse]],
                 logger: logging.Logger,
                 shutdown_event: Event,
                 device: str = "cuda"):
        super().__init__(name="ModelInference")
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = device
        self.logger = logger
        self.shutdown_event = shutdown_event
        self.daemon = True
    
    def run(self) -> None:
        """Main inference loop"""
        self.logger.info("Model inference worker started")
        while not self.shutdown_event.is_set():
            try:
                batch = self.input_queue.get(timeout=0.1)
                if batch is None:
                    continue
                
                start_time = time.time()
                responses = self._process_batch(batch)
                inference_time = time.time() - start_time
                
                self.logger.info(f"Model processed {len(batch.images)} images in {inference_time:.3f}s")
                self.output_queue.put(responses)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Model inference error: {e}", exc_info=True)
    
    def _process_batch(self, batch: ModelBatch) -> Dict[str, BatchResponse]:
        """Run inference and create responses for each client batch"""
        responses = {}
        
        # Get valid tensors and their indices
        try:
            tensor_batch, valid_indices = batch.get_tensor_batch()
            tensor_batch = tensor_batch.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(tensor_batch)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        except Exception as e:
            self.logger.error(f"Model inference failed: {e}")
            probabilities = None
            valid_indices = []
        
        # Create probability lookup
        prob_lookup = {}
        if probabilities is not None:
            for idx, prob_idx in enumerate(valid_indices):
                prob_lookup[prob_idx] = probabilities[idx]
        
        # Build responses for each client batch
        image_idx = 0
        for batch_id, client_images in batch.client_batches.items():
            scores = []
            for img in client_images:
                if img.error:
                    scores.append(ImageScore(
                        image_name=img.filename,
                        score1=0.0,
                        score2=0.0,
                        score3=0.0,
                        error=img.error
                    ))
                elif image_idx in prob_lookup:
                    probs = prob_lookup[image_idx]
                    scores.append(ImageScore(
                        image_name=img.filename,
                        score1=float(probs[0]),
                        score2=float(probs[1]),
                        score3=float(probs[2]),
                        error=None
                    ))
                else:
                    scores.append(ImageScore(
                        image_name=img.filename,
                        score1=0.0,
                        score2=0.0,
                        score3=0.0,
                        error="Inference failed"
                    ))
                image_idx += 1
            
            responses[batch_id] = BatchResponse(
                batch_id=batch_id,
                processing_time=batch.assembly_time,
                scores=scores
            )
        
        return responses

class ResponseManager(Thread):
    """Manages response delivery to clients"""
    
    def __init__(self,
                 response_queue: Queue[Dict[str, BatchResponse]],
                 logger: logging.Logger,
                 shutdown_event: Event):
        super().__init__(name="ResponseManager")
        self.response_queue = response_queue
        self.logger = logger
        self.shutdown_event = shutdown_event
        self.client_handlers: Dict[str, ConnectionHandler] = {}
        self.pending_responses: Dict[str, BatchResponse] = {}
        self.lock = Lock()
        self.daemon = True
    
    def register_handler(self, batch_id: str, handler: ConnectionHandler) -> None:
        """Register handler for a batch"""
        with self.lock:
            self.client_handlers[batch_id] = handler
            # Check if response already waiting
            if batch_id in self.pending_responses:
                response = self.pending_responses.pop(batch_id)
                try:
                    handler._send_response(response)
                except Exception as e:
                    self.logger.error(f"Failed to send pending response: {e}")
    
    def unregister_client(self, client_id: str) -> None:
        """Remove all handlers for a disconnected client"""
        with self.lock:
            to_remove = [bid for bid, h in self.client_handlers.items() 
                        if h.client_id == client_id]
            for bid in to_remove:
                del self.client_handlers[bid]
    
    def run(self) -> None:
        """Main response delivery loop"""
        self.logger.info("Response manager started")
        while not self.shutdown_event.is_set():
            try:
                responses = self.response_queue.get(timeout=0.1)
                if responses is None:
                    continue
                
                with self.lock:
                    for batch_id, response in responses.items():
                        if batch_id in self.client_handlers:
                            handler = self.client_handlers.pop(batch_id)
                            try:
                                handler._send_response(response)
                            except Exception as e:
                                self.logger.error(f"Failed to send response: {e}")
                        else:
                            # Store for later delivery
                            self.pending_responses[batch_id] = response
                            
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Response manager error: {e}", exc_info=True)

class MLBatchServer:
    """Main server coordinating all components"""
    
    def __init__(self,
                 host: str = "127.0.0.1",  # Default to localhost for testing
                 port: int = 5000,
                 model_path: str = "model/checkpoints/best_model_acc0.8328.pth",
                 transform_workers: int = 2,
                 max_batch_queue_size: int = 10,
                 model_batch_size: int = 40,
                 batch_timeout: float = 2.0):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.transform_workers = transform_workers
        self.max_batch_queue_size = max_batch_queue_size
        self.model_batch_size = model_batch_size
        self.batch_timeout = batch_timeout
        
        # Components
        self.server_socket: Optional[socket.socket] = None
        self.model: Optional[torch.nn.Module] = None
        self.batch_queue: Optional[BatchQueue] = None
        self.transform_queue: Optional[Queue] = None
        self.assembled_queue: Optional[Queue] = None
        self.model_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self.response_manager: Optional[ResponseManager] = None
        
        # Workers
        self.transform_workers_list: List[TransformWorker] = []
        self.batch_assembler: Optional[BatchAssembler] = None
        self.model_worker: Optional[ModelInferenceWorker] = None
        
        # Connection handlers
        self.connection_handlers: List[ConnectionHandler] = []
        
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = Event()
        self.accept_thread: Optional[Thread] = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def start(self) -> None:
        """Start the server and all components"""
        self.logger.info(f"Starting ML Batch Server on {self.host}:{self.port}")
        
        try:
            # Initialize components
            self._setup_queues()
            self._load_model()
            self._start_workers()
            self._setup_network()
            
            # Start accepting connections in a separate thread
            self.accept_thread = Thread(target=self._accept_connections, name="AcceptThread")
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            self.logger.info("Server started successfully")
            
            # Keep main thread alive
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}", exc_info=True)
            self.stop()
            raise
    
    def stop(self) -> None:
        """Gracefully stop the server"""
        self.logger.info("Stopping server...")
        self.shutdown_event.set()
        
        # Close server socket to stop accepting new connections
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Stop all connection handlers
        for handler in self.connection_handlers:
            handler.stop()
        
        # Wait for threads to finish
        threads_to_join = [
            self.accept_thread,
            self.response_manager,
            self.model_worker,
            self.batch_assembler
        ] + self.transform_workers_list
        
        for thread in threads_to_join:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        self.logger.info("Server stopped")
    
    def _setup_queues(self) -> None:
        """Initialize all queues"""
        self.batch_queue = BatchQueue(max_size=self.max_batch_queue_size)
        self.transform_queue = Queue()
        self.assembled_queue = Queue()
        self.model_queue = Queue()
        self.response_queue = Queue()
    
    def _setup_network(self) -> None:
        """Initialize server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.logger.info(f"Server listening on {self.host}:{self.port}")
    
    def _load_model(self) -> None:
        """Load ML model for inference"""
        self.model = load_for_inference(self.model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.logger.info("Model loaded on GPU")
        else:
            self.logger.warning("CUDA not available, using CPU for inference")
    
    def _start_workers(self) -> None:
        """Start all worker threads"""
        # Response manager
        self.response_manager = ResponseManager(
            self.response_queue, 
            self.logger,
            self.shutdown_event
        )
        self.response_manager.start()
        
        # Transform workers
        for i in range(self.transform_workers):
            worker = TransformWorker(
                worker_id=i,
                input_queue=self.transform_queue,
                output_queue=self.assembled_queue,
                logger=self.logger,
                shutdown_event=self.shutdown_event
            )
            worker.start()
            self.transform_workers_list.append(worker)
        
        # Batch assembler
        self.batch_assembler = BatchAssembler(
            input_queue=self.assembled_queue,
            output_queue=self.model_queue,
            logger=self.logger,
            shutdown_event=self.shutdown_event,
            max_batch_size=self.model_batch_size,
            timeout_seconds=self.batch_timeout
        )
        self.batch_assembler.start()
        
        # Model worker
        self.model_worker = ModelInferenceWorker(
            model=self.model,
            input_queue=self.model_queue,
            output_queue=self.response_queue,
            logger=self.logger,
            shutdown_event=self.shutdown_event,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_worker.start()
        
        # Queue transfer thread
        def transfer_batches():
            while not self.shutdown_event.is_set():
                batch = self.batch_queue.get(timeout=0.1)
                if batch:
                    self.transform_queue.put(batch)
        
        transfer_thread = Thread(target=transfer_batches, name="QueueTransfer")
        transfer_thread.daemon = True
        transfer_thread.start()
    
    def _accept_connections(self) -> None:
        """Main loop accepting client connections"""
        while not self.shutdown_event.is_set():
            try:
                self.server_socket.settimeout(0.5)
                client_socket, client_address = self.server_socket.accept()
                
                # Create handler for this connection
                handler = ConnectionHandler(
                    client_socket=client_socket,
                    client_address=client_address,
                    batch_queue=self.batch_queue,
                    response_manager=self.response_manager,
                    logger=self.logger
                )
                self.connection_handlers.append(handler)
                handler.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if not self.shutdown_event.is_set():
                    self.logger.error(f"Error accepting connection: {e}")

# ==================== Client Components ====================

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
    
    def __init__(self,
                 server_host: str = "127.0.0.1",
                 server_port: int = 5000,
                 reconnect_delay: float = 1.0,
                 max_reconnect_attempts: int = 5):
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
                self.logger.info(f"Connected to server at {self.server_host}:{self.server_port}")
                
                # Start response thread
                self.response_thread = Thread(target=self._receive_responses, name="ClientReceiver")
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
    
    def submit_batch(self,
                    images: List[Tuple[str, bytes]],
                    callback: Optional[ResponseCallback] = None) -> str:
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
    
    def submit_batch_sync(self,
                         images: List[Tuple[str, bytes]],
                         timeout: float = 30.0) -> BatchResponse:
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

# ==================== Test Functions ====================

def test_server():
    """Run the server"""
    server = MLBatchServer(
        host="127.0.0.1",
        port=5000,
        transform_workers=2,
        model_batch_size=40,
        batch_timeout=2.0
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        pass

def test_client():
    """Test client with sample images"""
    import os
    import glob
    
    client = MLBatchClient(server_host="127.0.0.1", server_port=5000)
    
    try:
        client.connect()
        
        # Load test images
        images = []
        
        # Try to find images in current directory
        image_files = glob.glob("*.jpg") + glob.glob("*.png")
        
        if not image_files:
            # Create dummy image data for testing
            print("No image files found, using dummy data")
            from PIL import Image
            import io
            
            for i in range(5):
                # Create a simple test image
                img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                images.append((f"test_image_{i}.jpg", buffer.getvalue()))
        else:
            # Load actual images
            for img_path in image_files[:5]:  # Limit to 5 images
                with open(img_path, 'rb') as f:
                    images.append((os.path.basename(img_path), f.read()))
                print(f"Loaded {img_path}")
        
        # Submit batch
        print(f"Submitting batch of {len(images)} images...")
        response = client.submit_batch_sync(images, timeout=10.0)
        
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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        test_client()
    else:
        print("Starting server (press Ctrl+C to stop)...")
        test_server()