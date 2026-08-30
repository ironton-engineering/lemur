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
    compute_capability: str = ""
    driver_version: str = ""
    supported: bool = True
    unsupported_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "memory_free_mib": self.memory_free_mib,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }


def list_gpus(*, include_unsupported: bool = False) -> list[GPU]:
    from scripts.system_probe import gpu_supported

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap,driver_version",
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
        if len(parts) < 7:
            continue
        try:
            total = int(float(parts[2]))
            used = int(float(parts[3]))
            free = int(float(parts[4]))
            supported, reason = gpu_supported(parts[1], parts[5])
            gpu = GPU(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mib=total,
                    memory_used_mib=used,
                    memory_free_mib=free,
                    compute_capability=parts[5],
                    driver_version=parts[6],
                    supported=supported,
                    unsupported_reason="" if supported else reason,
                )
            if include_unsupported or gpu.supported:
                gpus.append(gpu)
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
