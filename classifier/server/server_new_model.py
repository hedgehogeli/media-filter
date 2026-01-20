import logging
import signal
import socket
import time
from io import BytesIO
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Dict, List, Optional, Tuple, Union
import os
import torchvision.transforms.v2 as transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
from ultralytics import YOLO
import torch
from common_util import (
    BatchResponse,
    ClientBatch,
    ImageScore,
    ModelBatch,
    ProcessedImage,
    ProtocolDecoder,
    ProtocolEncoder,
    QueueFullResponse,
)
from PIL import Image
from torchvision import models
import torch.nn as nn


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-3])
        
    def forward(self, x):
        features = self.feature_extractor(x) # [B, 1024, H/16, W/16]
        return features
    
class YOLOv11(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = YOLO("model/yolov11l-face.pt").model
        # self.backbone = torch.nn.Sequential(*list(self.model.model.children())[:7])  # Stops after C3k2 (layer 6)
        self.feature_model = torch.nn.Sequential(*list(self.model.model.children())[:10])  # Stops after SPPF (layer 9)
        
    def forward(self, x):
        return self.feature_model(x)
    
class CLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # CLIP's final hidden state before projection (not the projection itself)
        self.clip_output_dim = self.clip_model.config.hidden_size 

    def forward(self, x):

        outputs = self.clip_model(**x)
        pooled_output = outputs.pooler_output  # shape: [batch_size, 512]
        return pooled_output

class BiggerClassifier(torch.nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()
        self.clip = CLIP() # CLIP outputs: [B, 768]
        self.yolo = YOLOv11() # YOLO outputs: [B, 512, 20, 20]
        self.resnet = ResNet() # ResNet outputs: [B, 1024, H/16, W/16]

        # Global average pooling for feature maps
        self.yolo_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.resnet_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # self.fc1 = torch.nn.Linear(768 + 512 + 1024, 2048)
        self.fc1 = torch.nn.Linear(768 + 512, 2048)
        self.activation1 = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.activation2 = torch.nn.GELU()
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(1024, output_dim)
        
    def forward(self, clip_inputs, img_tensor):
        clip_features = self.clip(clip_inputs)  # [B, 768]
        yolo_features = self.yolo(img_tensor)  # [B, 512, 20, 20]
        # resnet_features = self.resnet(img_tensor) # [B, 1024, _, _]

        # Pool YOLO features to [B, 512, 1, 1] then to [B, 512]
        yolo_features = self.yolo_pool(yolo_features).flatten(1)
        # resnet_features = self.resnet_pool(resnet_features).flatten(1)

        # combined_features = torch.cat([clip_features, yolo_features, resnet_features], dim=1)  # [B, 2304]
        combined_features = torch.cat([clip_features, yolo_features], dim=1)
        
        x = self.fc1(combined_features)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = self.fc3(x)
        
        return x
    
def load_for_inference(checkpoint_path, device='cuda'):
    """
    Load model for inference from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model in eval mode
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model for inference from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = BiggerClassifier(output_dim=3)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Model loaded")
    
    return model

CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


to_tensor = transforms.Compose([
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
])

yolo_val_transform = transforms.Compose([
    to_tensor,
    transforms.Resize(size=700, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), # Resize maintaining aspect ratio, then pad to square
    transforms.CenterCrop(640), 
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
])

clip_val_transform = transforms.Compose([ 
    to_tensor,
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
])


#####################################################

class ConnectionHandler(Thread):
    """Handles individual client connections"""

    def __init__(
        self,
        client_socket: socket.socket,
        client_address: Tuple[str, int],
        batch_queue: "BatchQueue",
        response_manager: "ResponseManager",
        logger: logging.Logger,
    ):
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


def transform_image(image: Image.Image):
    """Transform image using the provided transforms, return CPU tensor"""

    yolo_image = yolo_val_transform(image)
    clip_image = clip_val_transform(image)
    clip_image = CLIP_PROCESSOR(images=clip_image, return_tensors="pt", do_rescale=False)
    clip_image['pixel_values'] = clip_image['pixel_values'].squeeze(0)
    return clip_image, yolo_image


class TransformWorker(Thread):
    """Worker thread for image transformation"""

    def __init__(
        self,
        worker_id: int,
        input_queue: Queue[ClientBatch],
        output_queue: Queue[Tuple[ClientBatch, List[ProcessedImage]]],
        logger: logging.Logger,
        shutdown_event: Event,
    ):
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

                self.logger.info(
                    f"Worker {self.worker_id} processed batch {batch.batch_id} "
                    f"({len(batch.images)} images) in {transform_time:.3f}s"
                )

                self.output_queue.put((batch, processed))
            except Empty:
                continue
            except Exception as e:
                self.logger.error(
                    f"Transform worker {self.worker_id} error: {e}", exc_info=True
                )
    def _process_batch(self, batch: ClientBatch) -> List[ProcessedImage]:
        """Transform all images in a client batch"""
        processed = []
        for img_data in batch.images:
            try:
                # Open image from bytes
                image = Image.open(BytesIO(img_data.data))
                
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    # Paste the image on the white background using alpha channel as mask
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != 'RGB':
                    # Convert any other format to RGB
                    image = image.convert('RGB')
                
                # Apply transform
                clip_tensor, yolo_tensor = transform_image(image)
                processed.append(ProcessedImage(
                    filename=img_data.filename,
                    clip_tensor=clip_tensor,
                    yolo_tensor=yolo_tensor
                ))
            except Exception as e:
                self.logger.error(f"Failed to process {img_data.filename}: {e}")
                processed.append(ProcessedImage(
                    filename=img_data.filename,
                    error=f"Transform failed: {str(e)}"
                ))
        return processed


class BatchAssembler(Thread):
    """Assembles client batches into model batches"""

    def __init__(
        self,
        input_queue: Queue[Tuple[ClientBatch, List[ProcessedImage]]],
        output_queue: Queue[ModelBatch],
        logger: logging.Logger,
        shutdown_event: Event,
        max_batch_size: int = 40,
        timeout_seconds: float = 2.0,
    ):
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

    def _should_process_current_assembly(
        self, current_size: int, incoming_size: int
    ) -> bool:
        """Determine if current assembly should be processed"""
        return current_size > 0 and current_size + incoming_size > self.max_batch_size

    def _flush_assembly(self) -> None:
        """Create and queue model batch from current assembly"""
        if not self.current_assembly:
            return

        model_batch = self._create_model_batch()
        if model_batch:
            self.output_queue.put(model_batch)
            self.logger.info(
                f"Assembled model batch with {len(model_batch.images)} images "
                f"from {len(model_batch.batch_ids)} client batches"
            )

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
            assembly_time=time.time() - (self.assembly_start_time or time.time()),
        )


class ModelInferenceWorker(Thread):
    """Performs ML model inference on batches"""

    def __init__(
        self,
        model: torch.nn.Module,
        input_queue: Queue[ModelBatch],
        output_queue: Queue[Dict[str, BatchResponse]],
        logger: logging.Logger,
        shutdown_event: Event,
        device: str = "cuda",
    ):
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

                self.logger.info(
                    f"Model processed {len(batch.images)} images in {inference_time:.3f}s"
                )
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
            clip_batch, yolo_batch = tensor_batch 
            
            # Move to device
            clip_batch = {k: v.to(self.device) for k, v in clip_batch.items()} 
            yolo_batch = yolo_batch.to(self.device) 

            # Run inference
            with torch.no_grad():
                logits = self.model(clip_batch, yolo_batch)
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
                    scores.append(
                        ImageScore(
                            image_name=img.filename,
                            score1=0.0,
                            score2=0.0,
                            score3=0.0,
                            error=img.error,
                        )
                    )
                elif image_idx in prob_lookup:
                    probs = prob_lookup[image_idx]
                    scores.append(
                        ImageScore(
                            image_name=img.filename,
                            score1=float(probs[0]),
                            score2=float(probs[1]),
                            score3=float(probs[2]),
                            error=None,
                        )
                    )
                else:
                    scores.append(
                        ImageScore(
                            image_name=img.filename,
                            score1=0.0,
                            score2=0.0,
                            score3=0.0,
                            error="Inference failed",
                        )
                    )
                image_idx += 1

            responses[batch_id] = BatchResponse(
                batch_id=batch_id, processing_time=batch.assembly_time, scores=scores
            )

        return responses


class ResponseManager(Thread):
    """Manages response delivery to clients"""

    def __init__(
        self,
        response_queue: Queue[Dict[str, BatchResponse]],
        logger: logging.Logger,
        shutdown_event: Event,
    ):
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
            to_remove = [
                bid
                for bid, h in self.client_handlers.items()
                if h.client_id == client_id
            ]
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

    def __init__(
        self,
        model: torch.nn.Module,
        host: str = "127.0.0.1",  # Default to localhost for testing
        port: int = 5000,
        transform_workers: int = 2,
        max_batch_queue_size: int = 10,
        model_batch_size: int = 40,
        batch_timeout: float = 2.0,
    ):  

        self.host = host
        self.port = port
        self.model = model 
        self.transform_workers = transform_workers
        self.max_batch_queue_size = max_batch_queue_size
        self.model_batch_size = model_batch_size
        self.batch_timeout = batch_timeout

        # Components
        self.server_socket: Optional[socket.socket] = None
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

        if torch.cuda.is_available():
            model = model.cuda()
            self.logger.info("Model loaded on GPU")
        else:
            self.logger.warning("CUDA not available, using CPU for inference")

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
            self._start_workers()
            self._setup_network()

            # Start accepting connections in a separate thread
            self.accept_thread = Thread(
                target=self._accept_connections, name="AcceptThread"
            )
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
            self.batch_assembler,
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

    def _start_workers(self) -> None:
        """Start all worker threads"""
        # Response manager
        self.response_manager = ResponseManager(
            self.response_queue, self.logger, self.shutdown_event
        )
        self.response_manager.start()

        # Transform workers
        for i in range(self.transform_workers):
            worker = TransformWorker(
                worker_id=i,
                input_queue=self.transform_queue,
                output_queue=self.assembled_queue,
                logger=self.logger,
                shutdown_event=self.shutdown_event,
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
            timeout_seconds=self.batch_timeout,
        )
        self.batch_assembler.start()

        # Model worker
        self.model_worker = ModelInferenceWorker(
            model=self.model,
            input_queue=self.model_queue,
            output_queue=self.response_queue,
            logger=self.logger,
            shutdown_event=self.shutdown_event,
            device="cuda" if torch.cuda.is_available() else "cpu",
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
                    logger=self.logger,
                )
                self.connection_handlers.append(handler)
                handler.start()

            except socket.timeout:
                continue
            except Exception as e:
                if not self.shutdown_event.is_set():
                    self.logger.error(f"Error accepting connection: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('PIL').setLevel(logging.WARNING)



    model_path = "model/checkpoints/resnet_clip_yolo_mean_teacher_20251101_130004/checkpoint_epoch2_step113365_acc0.1185_20251102_103555.pth"
    model = load_for_inference(model_path)

    server = MLBatchServer(
        model=model,
        host="192.168.56.1", # "127.0.0.1", # "192.168.56.1",
        port=5000,
        transform_workers=3,
        model_batch_size=40,
        batch_timeout=2.0
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        pass