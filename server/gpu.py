import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GPU:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int = 0
    memory_free_mib: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
        }


def list_gpus() -> list[GPU]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus: list[GPU] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
            gpus.append(
                GPU(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mib=total,
                    memory_used_mib=used,
                    memory_free_mib=free,
                )
            )
        except ValueError:
            continue
    return gpus


def available_ram_mib(reserve_mib: int = 8192) -> float:
    """Return reclaimable system RAM after a desktop safety reserve."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = float(line.split()[1]) / 1024.0
                return max(0.0, available - float(reserve_mib))
    except (OSError, ValueError, IndexError):
        pass
    return 0.0
