# ml_batch_server.py
import socket
import struct
import json
import threading
import queue
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ImageScore:
    """Score for a single image"""
    image_name: str
    score1: float
    score2: float
    score3: float
    error: Optional[str] = None

@dataclass
class BatchRequest:
    """A batch of images to process"""
    batch_id: str
    images: List[Tuple[str, bytes]]  # [(filename, data), ...]
    client_socket: socket.socket  # Keep track of which client to respond to

@dataclass
class BatchResponse:
    """Response for a batch request"""
    batch_id: str
    scores: List[ImageScore]
    processing_time: float

class TransformTask:
    """Single image to transform"""
    def __init__(self, batch_id: str, image_name: str, image_data: bytes, index: int):
        self.batch_id = batch_id
        self.image_name = image_name
        self.image_data = image_data
        self.index = index
        self.transformed_data = None
        self.error = None

class MLBatchServer:
    def __init__(self, 
                 host: str = '192.168.56.1',
                 port: int = 5000,
                 model_func: Callable = None,
                 transform_func: Callable = None,
                 num_transform_workers: int = 2,
                 max_queue_size: int = 10,
                 max_model_batch_size: int = 40):
        
        self.host = host
        self.port = port
        self.model_func = model_func
        self.transform_func = transform_func
        self.num_transform_workers = num_transform_workers
        self.max_queue_size = max_queue_size
        self.max_model_batch_size = max_model_batch_size
        
        # Queues
        self.batch_queue = queue.Queue(maxsize=max_queue_size)
        self.model_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Track batches being processed
        self.active_batches = {}  # batch_id -> {tasks, completed_count, client_socket}
        self.batch_lock = threading.Lock()
        
        # Transform thread pool
        self.transform_executor = ThreadPoolExecutor(max_workers=num_transform_workers)
        
        # Server state
        self.running = True
        self.server_socket = None
        
    def start(self):
        """Start all server components"""
        # Start ML model thread
        model_thread = threading.Thread(target=self.model_worker, name="ModelWorker")
        model_thread.daemon = True
        model_thread.start()
        
        # Start batch dispatcher thread
        dispatcher_thread = threading.Thread(target=self.batch_dispatcher, name="BatchDispatcher")
        dispatcher_thread.daemon = True
        dispatcher_thread.start()
        
        # Start response sender thread
        response_thread = threading.Thread(target=self.response_sender, name="ResponseSender")
        response_thread.daemon = True
        response_thread.start()
        
        # Start accepting connections
        self.accept_connections()
    
    def accept_connections(self):
        """Main thread - accept client connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        logger.info(f"Server listening on {self.host}:{self.port}")
        
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")
                
                # Handle client in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket,),
                    name=f"ClientHandler-{address[0]}"
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
    
    def handle_client(self, client_socket: socket.socket):
        """Handle a single client connection"""
        try:
            while True:
                # Read batch header
                header = self.recv_all(client_socket, 12)
                if not header:
                    break
                
                num_images, batch_id_len = struct.unpack('!II', header[:8])
                
                # Read batch ID
                batch_id = self.recv_all(client_socket, batch_id_len).decode('utf-8')
                logger.info(f"Receiving batch {batch_id} with {num_images} images")
                
                # Read images
                images = []
                for i in range(num_images):
                    # Read image header
                    img_header = self.recv_all(client_socket, 8)
                    filename_len, image_len = struct.unpack('!II', img_header)
                    
                    # Read filename and image data
                    filename = self.recv_all(client_socket, filename_len).decode('utf-8')
                    image_data = self.recv_all(client_socket, image_len)
                    
                    images.append((filename, image_data))
                
                # Create batch request
                batch_request = BatchRequest(batch_id, images, client_socket)
                
                # Put in queue (will block if queue is full)
                self.batch_queue.put(batch_request)
                logger.info(f"Batch {batch_id} queued for processing")
                
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            client_socket.close()
    
    def batch_dispatcher(self):
        """Dispatch batches for transform processing"""
        while self.running:
            try:
                # Get batch from queue
                batch_request = self.batch_queue.get(timeout=1.0)
                
                # Create transform tasks
                tasks = []
                for i, (filename, image_data) in enumerate(batch_request.images):
                    task = TransformTask(batch_request.batch_id, filename, image_data, i)
                    tasks.append(task)
                
                # Track active batch
                with self.batch_lock:
                    self.active_batches[batch_request.batch_id] = {
                        'tasks': tasks,
                        'completed_count': 0,
                        'client_socket': batch_request.client_socket,
                        'start_time': time.time()
                    }
                
                # Submit transform tasks
                for task in tasks:
                    self.transform_executor.submit(self.transform_image, task)
                
                logger.info(f"Dispatched {len(tasks)} transform tasks for batch {batch_request.batch_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
    
    def transform_image(self, task: TransformTask):
        """Transform a single image"""
        try:
            if self.transform_func:
                task.transformed_data = self.transform_func(task.image_data)
            else:
                # Dummy transform - just pass through
                task.transformed_data = task.image_data
            
            logger.debug(f"Transformed {task.image_name} from batch {task.batch_id}")
            
        except Exception as e:
            task.error = f"Transform failed: {str(e)}"
            logger.error(f"Transform error for {task.image_name}: {e}")
        
        # Mark task complete and check if batch is ready
        with self.batch_lock:
            batch_info = self.active_batches.get(task.batch_id)
            if batch_info:
                batch_info['completed_count'] += 1
                
                # If all transforms complete, send to model
                if batch_info['completed_count'] == len(batch_info['tasks']):
                    self.model_queue.put(task.batch_id)
                    logger.info(f"Batch {task.batch_id} ready for model processing")
    
    def model_worker(self):
        """Process batches through ML model"""
        while self.running:
            try:
                # Get batch ID from queue
                batch_id = self.model_queue.get(timeout=1.0)
                
                with self.batch_lock:
                    batch_info = self.active_batches.get(batch_id)
                    if not batch_info:
                        continue
                    
                    tasks = batch_info['tasks']
                    start_time = batch_info['start_time']
                
                # Prepare data for model
                valid_images = []
                image_indices = []
                
                for i, task in enumerate(tasks):
                    if task.transformed_data is not None and task.error is None:
                        valid_images.append(task.transformed_data)
                        image_indices.append(i)
                
                # Run model inference
                scores = []
                if valid_images and self.model_func:
                    try:
                        # Process in chunks if needed
                        all_results = []
                        for i in range(0, len(valid_images), self.max_model_batch_size):
                            chunk = valid_images[i:i + self.max_model_batch_size]
                            chunk_results = self.model_func(chunk)
                            all_results.extend(chunk_results)
                        
                        # Map results back to original indices
                        result_map = {image_indices[i]: all_results[i] for i in range(len(all_results))}
                        
                    except Exception as e:
                        logger.error(f"Model inference error: {e}")
                        result_map = {}
                else:
                    result_map = {}
                
                # Create response with scores
                for i, task in enumerate(tasks):
                    if task.error:
                        # Failed transform
                        score = ImageScore(task.image_name, 0.0, 0.0, 0.0, task.error)
                    elif i in result_map:
                        # Successful inference
                        s1, s2, s3 = result_map[i]
                        score = ImageScore(task.image_name, s1, s2, s3)
                    else:
                        # No model func or other error
                        score = ImageScore(task.image_name, 0.0, 0.0, 0.0, "Model processing failed")
                    
                    scores.append(score)
                
                # Create response
                processing_time = time.time() - start_time
                response = BatchResponse(batch_id, scores, processing_time)
                
                # Queue response with client socket
                with self.batch_lock:
                    client_socket = batch_info['client_socket']
                    self.response_queue.put((response, client_socket))
                    
                    # Clean up
                    del self.active_batches[batch_id]
                
                logger.info(f"Batch {batch_id} processed in {processing_time:.2f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Model worker error: {e}")
                traceback.print_exc()
    
    def response_sender(self):
        """Send responses back to clients"""
        while self.running:
            try:
                response, client_socket = self.response_queue.get(timeout=1.0)
                
                # Serialize response
                response_dict = {
                    'batch_id': response.batch_id,
                    'processing_time': response.processing_time,
                    'scores': [asdict(score) for score in response.scores]
                }
                
                json_data = json.dumps(response_dict).encode('utf-8')
                
                # Send response
                header = struct.pack('!I', len(response.batch_id.encode('utf-8')))
                header += response.batch_id.encode('utf-8')
                header += struct.pack('!I', len(json_data))
                
                client_socket.send(header)
                client_socket.send(json_data)
                
                logger.info(f"Sent response for batch {response.batch_id}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Response sender error: {e}")
    
    def recv_all(self, sock: socket.socket, length: int) -> bytes:
        """Receive exactly 'length' bytes"""
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        self.transform_executor.shutdown(wait=True)


# Example usage with dummy functions
def dummy_transform(image_data: bytes) -> bytes:
    """Dummy transform function - replace with actual torchvision transforms"""
    # In reality, this would:
    # 1. Decode image bytes to PIL/tensor
    # 2. Apply torchvision transforms
    # 3. Return transformed tensor/bytes
    return image_data

def dummy_model(image_batch: List[bytes]) -> List[Tuple[float, float, float]]:
    """Dummy model function - replace with actual ML model"""
    # In reality, this would:
    # 1. Convert batch to tensor
    # 2. Run through model
    # 3. Return scores
    import random
    return [(random.random(), random.random(), random.random()) for _ in image_batch]

if __name__ == "__main__":
    server = MLBatchServer(
        model_func=dummy_model,
        transform_func=dummy_transform,
        num_transform_workers=2,
        max_queue_size=10,
        max_model_batch_size=40
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop()

# Import your actual ML model and transforms
import torchvision.transforms as T
from your_model import load_model

# Define transform function
def transform_func(image_bytes):
    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Apply transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    return tensor

# Define model function
model = load_model()
def model_func(image_tensors):
    # Stack tensors and run inference
    batch = torch.stack(image_tensors)
    with torch.no_grad():
        outputs = model(batch)
    
    # Return list of (score1, score2, score3) tuples
    return [(o[0].item(), o[1].item(), o[2].item()) for o in outputs]

# Start server
server = MLBatchServer(
    model_func=model_func,
    transform_func=transform_func,
    num_transform_workers=4,  # Experiment with this
    max_queue_size=10
)
server.start()