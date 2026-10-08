
import subprocess
from typing import Literal

import psutil
from pydantic import BaseModel, Field
from mcp.server.mcpserver import MCPServer

SERVER_NAME = "BrainBox"
# SERVER_NAME = "my gorgeous notebook"

mcp = MCPServer(f"{SERVER_NAME} Monitor",
                instructions=f"Monitor RAM and NVIDIA GPU memory on {SERVER_NAME}.",)


# ---------- MCP TOOLS ----------

@mcp.tool()
def hello() -> str:
    """Return a greeting from BrainBox."""
    return "hello from mcp"

# ---------- OUTPUT SCHEMA ----------

class MemoryInfo(BaseModel):
    type: Literal["RAM", "VRAM"]
    available: bool
    gpu_id: int | None = None
    total_gib: float | None = None
    used_gib: float | None = None
    free_gib: float | None = None
    percent: float | None = Field(default=None, ge=0, le=100)
    message: str | None = None


# ---------- RESPONSE ----------

def response(kind, gpu_id=None, values=None, message=None) -> MemoryInfo:
    return MemoryInfo(
        type=kind,
        available=values is not None,
        gpu_id=gpu_id,
        **dict(zip(
            ("total_gib", "used_gib", "free_gib", "percent"),
            values or [None] * 4,
        )),
        message=message,
    )



def get_memory_info(
    type: Literal["RAM", "VRAM"],
    gpu_id: int = -1,
) -> MemoryInfo:
    
    # RAM
    if type == "RAM":
        m = psutil.virtual_memory()
        total, used, free = m.total, m.total - m.available, m.available
        return response("RAM", values=[
            round(total / 2**30, 2),
            round(used / 2**30, 2),
            round(free / 2**30, 2),
            round(used / total * 100, 2) if total else 0,
        ])

    # VRAM: query all NVIDIA GPUs
    if gpu_id < -1:
        return response("VRAM", gpu_id, message="Invalid GPU ID")

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.PIPE,
        )

        gpus = {
            int(i): (int(t), int(u), int(f))
            for i, t, u, f in (
                map(str.strip, line.split(","))
                for line in output.splitlines()
            )
        }

    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return response(
            "VRAM", gpu_id,
            message=f"GPU information unavailable: {e}",
        )

    # Select one GPU or aggregate all
    if not gpus:
        return response("VRAM", gpu_id, message="No NVIDIA GPUs detected")

    if gpu_id != -1 and gpu_id not in gpus:
        return response("VRAM", gpu_id, message=f"GPU {gpu_id} not found")

    selected = gpus.values() if gpu_id == -1 else [gpus[gpu_id]]
    total, used, free = map(sum, zip(*selected))

    return response("VRAM", gpu_id, values=[
        round(total / 1024, 2),
        round(used / 1024, 2),
        round(free / 1024, 2),
        round(used / total * 100, 2) if total else 0,
    ])

get_memory_info.__doc__ = f"""Get system RAM or NVIDIA GPU memory usage on a machine called {SERVER_NAME}.

    Args:
        type: RAM or VRAM.
        gpu_id: GPU index (0, 1, ...); -1 sums all GPUs.
                Ignored for RAM.
    """

mcp.tool()(get_memory_info)


# ---------- SERVER ----------

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        streamable_http_path="/mcp",
    )
