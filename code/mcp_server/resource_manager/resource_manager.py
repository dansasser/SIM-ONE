import os
import psutil
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    A simple resource manager to monitor CPU and memory usage.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    @contextmanager
    def profile(self, name: str = "operation"):
        """
        A context manager to profile the CPU and memory usage of a block of code.

        Args:
            name: The name of the operation being profiled.

        Yields:
            A dictionary to which the resource usage metrics will be added.
        """
        metrics = {}

        # Get initial CPU and memory usage
        mem_before = self.process.memory_info().rss
        cpu_before = self.process.cpu_times()

        try:
            yield metrics
        finally:
            # Get final CPU and memory usage
            mem_after = self.process.memory_info().rss
            cpu_after = self.process.cpu_times()

            # Calculate the difference
            metrics['cpu_user_time'] = cpu_after.user - cpu_before.user
            metrics['cpu_system_time'] = cpu_after.system - cpu_before.system
            metrics['cpu_total_time'] = (cpu_after.user - cpu_before.user) + \
                                        (cpu_after.system - cpu_before.system)
            metrics['mem_usage_bytes'] = mem_after - mem_before

            logger.info(
                f"Resource profile for '{name}': "
                f"CPU Time={metrics['cpu_total_time']:.4f}s, "
                f"Memory Usage={metrics['mem_usage_bytes']/1024/1024:.2f}MB"
            )

if __name__ == '__main__':
    # Example Usage
    import time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    rm = ResourceManager()

    logger.info("Profiling a sample operation...")
    with rm.profile("sample_heavy_computation") as metrics:
        # Simulate some work
        _ = [i*i for i in range(10**6)]
        time.sleep(0.5)

    logger.info(f"Collected metrics: {metrics}")

    logger.info("\nProfiling a memory-intensive operation...")
    with rm.profile("sample_memory_allocation") as metrics:
        # Simulate memory allocation
        a = ' ' * (10 * 1024 * 1024) # allocate 10MB
        del a

    logger.info(f"Collected metrics: {metrics}")
