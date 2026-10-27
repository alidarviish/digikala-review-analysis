"""Keep long runs from filling the memory of the machine.

Before a heavy step, a script calls require_free_memory. If there is not enough room it
stops with a clear message instead of pushing the system into swap or freezing it.
"""

from pathlib import Path

MEMINFO = Path("/proc/meminfo")
PROCESS_STATUS = Path("/proc/self/status")


def available_memory_gb(meminfo: Path = MEMINFO) -> float:
    """Memory the system can still give out without swapping (MemAvailable), in GB."""
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2  # the value is in kB
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def require_free_memory(needed_gb: float, what: str) -> None:
    available = available_memory_gb()
    if available < needed_gb:
        raise SystemExit(
            f"Stopped before {what}: {available:.1f} GB free, {needed_gb:.1f} GB needed. "
            "Close other programs and run the script again; finished work is kept."
        )


def peak_memory_gb(status: Path = PROCESS_STATUS) -> float:
    """Largest memory this process has used so far (VmHWM), in GB.

    Not resource.getrusage: on Linux its peak survives exec, so a child process started
    from a big parent reports the parent's peak instead of its own.
    """
    for line in status.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024**2  # the value is in kB
    raise RuntimeError("VmHWM not found in /proc/self/status")


def peak_memory_gb_if_known(status: Path = PROCESS_STATUS) -> float | None:
    """peak_memory_gb rounded to 0.01, or None where the system does not report it.

    The sandbox that runs molab notebooks has no VmHWM line; a result file then records the
    peak as unknown instead of the run stopping.
    """
    try:
        return round(peak_memory_gb(status), 2)
    except RuntimeError:
        return None
