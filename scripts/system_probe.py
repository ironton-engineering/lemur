#!/usr/bin/env python3
"""System checks shared by the Lemur installer and support command."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_UBUNTU = {"22.04", "24.04"}
MIN_DRIVER = (570, 26)
MIN_COMPUTE_CAPABILITY = (8, 6)
RTX_SERIES_RE = re.compile(r"\b(?:GeForce\s+)?RTX\s+(30|40|50)\d{2}\b", re.I)


@dataclass(frozen=True)
class GPUReport:
    index: int
    name: str
    memory_total_mib: int
    compute_capability: str
    driver_version: str
    supported: bool
    reason: str


def version_tuple(value: str) -> tuple[int, ...]:
    values = re.findall(r"\d+", value)
    return tuple(int(item) for item in values)


def read_os_release(path: str | Path = "/etc/os-release") -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path(path).read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def gpu_supported(name: str, compute_capability: str) -> tuple[bool, str]:
    capability = version_tuple(compute_capability)
    if capability and capability < MIN_COMPUTE_CAPABILITY:
        return False, f"compute capability {compute_capability} is below 8.6"
    if not RTX_SERIES_RE.search(name):
        return False, "GPU is not an NVIDIA RTX 30-, 40-, or 50-series device"
    if not capability:
        return False, "compute capability is not available"
    return True, "supported"


def parse_nvidia_smi(text: str) -> list[GPUReport]:
    reports: list[GPUReport] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            memory = int(float(parts[2]))
        except ValueError:
            continue
        supported, reason = gpu_supported(parts[1], parts[3])
        reports.append(
            GPUReport(
                index=index,
                name=parts[1],
                memory_total_mib=memory,
                compute_capability=parts[3],
                driver_version=parts[4],
                supported=supported,
                reason=reason,
            )
        )
    return reports


def query_gpus() -> tuple[list[GPUReport], str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return [], "nvidia-smi is not installed"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi did not respond"
    if result.returncode != 0:
        message = result.stderr.strip() or "nvidia-smi failed"
        return [], message
    reports = parse_nvidia_smi(result.stdout)
    if not reports:
        return [], "nvidia-smi did not return GPU data"
    return reports, None


def cuda_architectures(gpus: list[GPUReport]) -> list[str]:
    """Return CMake CUDA architecture values for supported GPUs."""
    architectures = {
        gpu.compute_capability.replace(".", "")
        for gpu in gpus
        if gpu.supported and re.fullmatch(r"\d+\.\d+", gpu.compute_capability)
    }
    return sorted(architectures, key=int)


def disk_free_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path).expanduser()).free / (1024**3)


def inspect_system(
    *,
    os_release_path: str | Path = "/etc/os-release",
    install_root: str | Path | None = None,
) -> dict:
    os_data = read_os_release(os_release_path)
    machine = platform.machine()
    gpus, gpu_error = query_gpus()
    supported = [gpu for gpu in gpus if gpu.supported]
    ignored = [gpu for gpu in gpus if not gpu.supported]
    driver_ok = bool(supported) and all(
        version_tuple(gpu.driver_version) >= MIN_DRIVER for gpu in supported
    )
    root = Path(install_root or os.environ.get("LEMUR_INSTALL_ROOT", "~/.local/share/llm-hub")).expanduser()
    disk_path = root.parent if root.parent.exists() else Path.home()
    errors: list[str] = []
    if os_data.get("ID") != "ubuntu" or os_data.get("VERSION_ID") not in SUPPORTED_UBUNTU:
        errors.append("Lemur supports Ubuntu 22.04 and 24.04 only")
    if machine != "x86_64":
        errors.append("Lemur supports x86-64 only")
    if gpu_error:
        errors.append(gpu_error)
    elif not supported:
        errors.append("No supported NVIDIA RTX 30-, 40-, or 50-series GPU was found")
    if supported and not driver_ok:
        errors.append("NVIDIA driver 570.26 or newer is required")
    free_gib = disk_free_gib(disk_path)
    if free_gib < 15:
        errors.append("At least 15 GiB of free disk space is required")
    return {
        "ok": not errors,
        "os_id": os_data.get("ID", "unknown"),
        "os_version": os_data.get("VERSION_ID", "unknown"),
        "machine": machine,
        "minimum_driver": ".".join(str(item) for item in MIN_DRIVER),
        "driver_ok": driver_ok,
        "free_disk_gib": round(free_gib, 1),
        "supported_gpus": [asdict(gpu) for gpu in supported],
        "ignored_gpus": [asdict(gpu) for gpu in ignored],
        "errors": errors,
    }


def print_report(report: dict) -> None:
    print(f"Ubuntu: {report['os_version']} ({report['machine']})")
    for gpu in report["supported_gpus"]:
        print(
            f"GPU {gpu['index']}: {gpu['name']} "
            f"({gpu['memory_total_mib']} MiB, compute {gpu['compute_capability']})"
        )
    for gpu in report["ignored_gpus"]:
        print(f"Ignored GPU {gpu['index']}: {gpu['name']} ({gpu['reason']})")
    print(f"Free disk space: {report['free_disk_gib']} GiB")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
    else:
        print("System check: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a system for Lemur")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--cuda-architectures",
        action="store_true",
        help="print supported CMake CUDA architecture values",
    )
    parser.add_argument("--os-release", default="/etc/os-release", help=argparse.SUPPRESS)
    parser.add_argument("--install-root", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = inspect_system(
        os_release_path=args.os_release,
        install_root=args.install_root,
    )
    if args.cuda_architectures:
        print(";".join(cuda_architectures([
            GPUReport(**gpu) for gpu in report["supported_gpus"]
        ])))
    elif args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
