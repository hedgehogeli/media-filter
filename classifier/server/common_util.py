import json
import socket
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch


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
    clip_tensor: Optional[torch.Tensor] = None
    yolo_tensor: Optional[torch.Tensor] = None
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
    """Batch of images from client (1-20 images)"""

    batch_id: str
    images: List[ImageData]
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None

    def validate(self) -> None:
        """Validate batch constraints"""
        if not (1 <= len(self.images) <= 20):
            raise ValueError(f"Batch must contain 1-20 images, got {len(self.images)}")
        for img in self.images:
            img.validate()


@dataclass
class ModelBatch:
    """Assembled batch for model inference (up to 40 images)"""

    batch_ids: List[str]  # Client batch IDs included
    images: List[ProcessedImage]
    client_batches: Dict[str, List[ProcessedImage]]  # Maintain client batch grouping
    assembly_time: float = 0.0

    def get_tensor_batch(self) -> Tuple[Tuple[Dict, torch.Tensor], List[int]]:
        """
        Get batched tensors from all valid images.
        
        Returns:
            tuple: ((clip_batch_dict, yolo_batch_tensor), valid_indices)
        """
        clip_tensors = []
        yolo_tensors = []
        valid_indices = []
        
        for idx, img in enumerate(self.images):
            if img.error is None and img.clip_tensor is not None and img.yolo_tensor is not None:
                clip_tensors.append(img.clip_tensor)
                yolo_tensors.append(img.yolo_tensor)
                valid_indices.append(idx)
        
        if not clip_tensors:
            raise ValueError("No valid images in batch")
        
        # Stack CLIP pixel_values (assuming they're dicts with 'pixel_values' key)
        clip_batch = {
            'pixel_values': torch.stack([t['pixel_values'] for t in clip_tensors])
        }
        
        # Stack YOLO tensors
        yolo_batch = torch.stack(yolo_tensors)
        
        return (clip_batch, yolo_batch), valid_indices


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
