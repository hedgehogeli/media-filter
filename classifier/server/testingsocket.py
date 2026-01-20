# import socket
# import struct
# import os
# import queue
# import threading
# import time

# class ImageQueueServer:
#     def __init__(self, host='192.168.56.1', port=5000):
#         self.host = host
#         self.port = port
#         self.image_queue = queue.Queue()
#         self.server_socket = None
        
#     def start(self):
#         self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         self.server_socket.bind((self.host, self.port))
#         self.server_socket.listen(5)
#         print(f"Server listening on {self.host}:{self.port}")
        
#         # Start image processor thread
#         processor_thread = threading.Thread(target=self.process_images)
#         processor_thread.daemon = True
#         processor_thread.start()
        
#         # Accept connections
#         while True:
#             client_socket, address = self.server_socket.accept()
#             print(f"Connection from {address}")
#             client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
#             client_thread.daemon = True
#             client_thread.start()
    
#     def handle_client(self, client_socket):
#         try:
#             while True:
#                 # Receive header: 4 bytes for filename length, 4 bytes for data length
#                 header = self.recv_all(client_socket, 8)
#                 if not header:
#                     break
                    
#                 filename_len, data_len = struct.unpack('!II', header)
                
#                 # Receive filename
#                 filename = self.recv_all(client_socket, filename_len).decode('utf-8')
                
#                 # Receive image data
#                 image_data = self.recv_all(client_socket, data_len)
                
#                 # Add to queue
#                 self.image_queue.put((filename, image_data))
#                 print(f"Queued image: {filename} ({data_len} bytes)")
                
#                 # Send acknowledgment
#                 client_socket.send(b'OK')
                
#         except Exception as e:
#             print(f"Client error: {e}")
#         finally:
#             client_socket.close()
    
#     def recv_all(self, sock, length):
#         """Receive exactly 'length' bytes"""
#         data = b''
#         while len(data) < length:
#             packet = sock.recv(length - len(data))
#             if not packet:
#                 return None
#             data += packet
#         return data
    
#     def process_images(self):
#         """Process images from the queue"""
#         while True:
#             try:
#                 filename, image_data = self.image_queue.get(timeout=1)
#                 print(f"Processing {filename}...")
                
#                 # Save to processed folder
#                 output_path = os.path.join('processed', filename)
#                 os.makedirs('processed', exist_ok=True)
                
#                 with open(output_path, 'wb') as f:
#                     f.write(image_data)
                
#                 # Here you would do actual image processing
#                 # For example: resize, convert, apply filters, etc.
                
#                 print(f"Processed {filename}")
                
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f"Processing error: {e}")

# test_server.py
import socket
import struct
import os
import queue
import threading
import time

class ImageQueueServer:
    def __init__(self, host='192.168.56.1', port=5000):
        self.host = host
        self.port = port
        self.image_queue = queue.Queue()
        self.server_socket = None
        self.running = True
        
    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        
        # Start image processor thread
        processor_thread = threading.Thread(target=self.process_images)
        processor_thread.daemon = True
        processor_thread.start()
        
        # Accept connections
        while self.running:
            try:
                self.server_socket.settimeout(1.0)  # Allow checking self.running
                client_socket, address = self.server_socket.accept()
                print(f"Connection from {address}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Server error: {e}")
    
    def handle_client(self, client_socket):
        try:
            while True:
                # Receive header: 4 bytes for filename length, 4 bytes for data length
                header = self.recv_all(client_socket, 8)
                if not header:
                    break
                    
                filename_len, data_len = struct.unpack('!II', header)
                
                # Receive filename
                filename = self.recv_all(client_socket, filename_len).decode('utf-8')
                
                # Receive image data
                image_data = self.recv_all(client_socket, data_len)
                
                # Add to queue
                self.image_queue.put((filename, image_data))
                print(f"Queued image: {filename} ({data_len} bytes)")
                
                # Send acknowledgment
                client_socket.send(b'OK')
                
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()
    
    def recv_all(self, sock, length):
        """Receive exactly 'length' bytes"""
        data = b''
        while len(data) < length:
            packet = sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def process_images(self):
        """Process images from the queue"""
        while self.running:
            try:
                filename, image_data = self.image_queue.get(timeout=1)
                print(f"Processing {filename}...")
                
                # Save to processed folder
                output_path = os.path.join('processed', filename)
                os.makedirs('processed', exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                
                print(f"Saved {filename} to {output_path}")
                print(f"File size: {len(image_data)} bytes")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

# Main execution
if __name__ == "__main__":
    print("Starting Image Queue Server...")
    print("Press Ctrl+C to stop")
    
    server = ImageQueueServer(host='192.168.56.1', port=5000)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
        print("Server stopped.")