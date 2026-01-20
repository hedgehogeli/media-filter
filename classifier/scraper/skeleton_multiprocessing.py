# MULTIPROCESSING SKELETON

import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict


def setup_worker_logger(worker_id: int, log_dir: Path) -> logging.Logger:
    """Set up a logger for each worker that writes to its own subdirectory."""
    worker_dir = log_dir / f"worker_{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.DEBUG)

    # Create file handler
    log_file = worker_dir / f"worker_{worker_id}.log"
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
    log_dir: Path,
):
    """Worker function that processes jobs from the queue."""
    # Set up logging for this worker
    logger = setup_worker_logger(worker_id, log_dir)
    logger.info(f"Worker {worker_id} started")

    # Create worker subdirectory for any additional files
    worker_dir = log_dir / f"worker_{worker_id}"

    try:
        while not shutdown_event.is_set():
            try:
                # Try to get a job with timeout to periodically check shutdown
                job = job_queue.get(timeout=1.0)

                if job is None:  # Poison pill
                    logger.info(f"Worker {worker_id} received poison pill")
                    break

                logger.info(f"Worker {worker_id} processing job: {job}")

                # Simulate some work
                result = process_job(job, worker_id, worker_dir)

                # Put result in result queue
                result_queue.put(result)
                logger.info(f"Worker {worker_id} completed job: {job}")

            except queue.Empty:
                # No job available, continue to check shutdown
                continue
            except Exception as e:
                logger.error(
                    f"Worker {worker_id} error processing job: {e}", exc_info=True
                )

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} interrupted")
    finally:
        logger.info(f"Worker {worker_id} shutting down")


def process_job(
    job: Dict[str, Any], worker_id: int, worker_dir: Path
) -> Dict[str, Any]:
    """Process a single job and return results."""
    # Simulate some work
    time.sleep(0.5)

    # Example: write some output to worker directory
    output_file = worker_dir / f"job_{job.get('id', 'unknown')}_output.txt"
    with open(output_file, "w") as f:
        f.write(f"Processed by worker {worker_id}\n")
        f.write(f"Job data: {job}\n")

    # Return result dictionary
    return {
        "job_id": job.get("id"),
        "worker_id": worker_id,
        "status": "completed",
        "output_file": str(output_file),
        "timestamp": time.time(),
    }


def result_collector(
    result_queue: mp.Queue, shutdown_event: mp.Event, logger: logging.Logger
):
    """Periodically collect results from the result queue."""
    accumulated_results = []

    while not shutdown_event.is_set() or not result_queue.empty():
        try:
            # Try to get results with timeout
            result = result_queue.get(timeout=0.5)
            accumulated_results.append(result)
            logger.info(f"Collected result: {result}")

            # Process accumulated results periodically
            if len(accumulated_results) >= 5:
                process_accumulated_results(accumulated_results, logger)
                accumulated_results.clear()

        except queue.Empty:
            # Process any remaining results
            if accumulated_results:
                process_accumulated_results(accumulated_results, logger)
                accumulated_results.clear()

    # Process any final results
    if accumulated_results:
        process_accumulated_results(accumulated_results, logger)


def process_accumulated_results(results: list, logger: logging.Logger):
    """Process a batch of accumulated results."""
    logger.info(f"Processing {len(results)} accumulated results")
    # Add your processing logic here
    for result in results:
        logger.debug(f"Result: {result}")


def signal_handler(signum, frame, shutdown_event: mp.Event, logger: logging.Logger):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


def main():
    """Main function that manages the job queue and workers."""
    # Set up logging for main process
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Number of worker processes
    n_workers = mp.cpu_count() - 1 or 1
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
            args=(i, job_queue, result_queue, shutdown_event, log_dir),
        )
        worker.start()
        workers.append(worker)
        logger.info(f"Started worker {i}")

    # Start result collector in a separate thread
    import threading

    collector_thread = threading.Thread(
        target=result_collector, args=(result_queue, shutdown_event, logger)
    )
    collector_thread.start()

    try:
        # Enqueue some example jobs
        for i in range(20):
            if shutdown_event.is_set():
                break

            job = {"id": i, "data": f"job_data_{i}", "timestamp": time.time()}
            job_queue.put(job)
            logger.info(f"Enqueued job {i}")
            time.sleep(0.1)  # Simulate job creation delay

        # Wait for all jobs to be processed
        logger.info("All jobs enqueued, waiting for completion...")

        # Send poison pills to workers
        for _ in range(n_workers):
            job_queue.put(None)

        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=30)
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
