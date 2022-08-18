import logging
import os

import psutil

logger = logging.getLogger(__name__)


def mem_based_cpu_suggestion():
    """
    Return an estimate of the number of clone processes
    that would fit in the currently available memory.
    """

    # Get the resident size of current process
    pid = os.getpid()
    process = psutil.Process(pid)
    rss_mem_usage = process.memory_info().rss

    # Get the free memory
    free_mem = psutil.virtual_memory()[4]

    # Calculate how many processes would fit using a 10% safety margin.
    n = int(free_mem // (rss_mem_usage * 1.1)) + 1  # Count current process

    logger.info(
        f"Current process usage {rss_mem_usage}"
        f" and {free_mem} free memory, could fit {n} processes in RAM."
    )

    return n


def physical_core_cpu_suggestion():
    """
    Return the physical cores.
    """

    n = psutil.cpu_count(logical=False)
    logger.info(f"Found {n} physical cores")
    return n


def virtual_core_cpu_suggestion():
    """
    Return the virtual cores.
    """

    n = psutil.cpu_count(logical=True)
    logger.info(f"Found {n} logical cores")
    return n


def num_procs_suggestion():
    """
    Resolve and return number of processors to use for multiprocessing.

    Query memory and cpu, then makes suggestion.
    """
    suggestions = {
        "memory": mem_based_cpu_suggestion(),
        "physical cores": physical_core_cpu_suggestion(),
    }
    k = sorted(suggestions, key=suggestions.get)[0]
    n = suggestions[k]

    logger.info(f"Suggesting {n} processors based on {k}.")
    return n
