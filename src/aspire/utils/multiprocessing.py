import logging
import os

import psutil

logger = logging.getLogger(__name__)


def mem_based_cpu_suggestion():
    pid = os.getpid()
    process = psutil.Process(pid)
    rss_mem_usage = process.memory_info().rss

    free_mem = psutil.virtual_memory()[4]

    n = free_mem // (rss_mem_usage * 1.1)

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


def get_num_multi_procs(n):
    """
    Resolve and return number of processors to use for multiprocessing.

    If n is 'auto', query memory and cpu, then makes suggestion.
    Developers should ensure n=0, n=None, n=False runs code serially without Pool.
    Otherwise, n is expected to be an integer and will pass through.
    """
    if isinstance(n, int):
        return n
    elif n is None or n is False:
        return 0
    elif n.lower() == "auto":
        suggestions = {
            "mememory": mem_based_cpu_suggestion(),
            "physical cores": physical_core_cpu_suggestion(),
        }
        k = sorted(suggestions, key=suggestions.get)[0]
        n = suggestions[k]

        logger.info(f"Suggesting {n} multi processor based on {k}.")

        return n
    else:
        raise ValueError(f"Unable to parse {n}, try an integer or 'auto'")
