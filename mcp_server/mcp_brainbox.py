
import subprocess

import psutil
from mcp.server.mcpserver import MCPServer


mcp = MCPServer(
    "BrainBox Monitor",
    instructions="Read-only tools for monitoring RAM and NVIDIA VRAM.",
)


@mcp.tool()
def get_ram_info() -> dict:
    """Return total, available and used system RAM in GiB."""

    mem = psutil.virtual_memory()
    gib = 1024 ** 3

    return {
        "total_gib": round(mem.total / gib, 2),
        "available_gib": round(mem.available / gib, 2),
        "used_gib": round(mem.used / gib, 2),
        "percent": mem.percent,
    }


@mcp.tool()
def get_vram_info() -> dict:
    """Return NVIDIA GPU VRAM usage, if NVIDIA GPUs are available."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "message": f"NVIDIA GPU information unavailable: {exc}",
        }

    gpus = []

    for line in result.stdout.strip().splitlines():
        name, total, used, free = (
            part.strip() for part in line.split(",")
        )

        gpus.append({
            "name": name,
            "total_mib": int(total),
            "used_mib": int(used),
            "free_mib": int(free),
        })

    return {
        "available": bool(gpus),
        "gpus": gpus,
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        streamable_http_path="/mcp",
    )
