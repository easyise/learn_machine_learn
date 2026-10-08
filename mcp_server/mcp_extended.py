import subprocess
import urllib.request
from pathlib import Path

from mcp.server import MCPServer


# ============================================================
# Configuration
# ============================================================

DEMO_DIR = Path.home() / "mcp-demo"

NOTE_FILE = DEMO_DIR / "note.txt"

TIME_SERVICE_FILE = DEMO_DIR / "time_server.py"
TIME_SERVICE_HOST = "0.0.0.0"
TIME_SERVICE_PORT = 8080

PID_FILE = DEMO_DIR / ".time_service.pid"
LOG_FILE = DEMO_DIR / "time_service.log"


# ============================================================
# MCP server
# ============================================================

mcp = MCPServer(
    "TimeService Demo",
    instructions=(
        "Tools for working with the TimeService MCP demo environment."
    ),
)


# ============================================================
# NOTE TOOLS
# ============================================================

@mcp.tool()
def get_note() -> dict:
    """
    Read the current demo note stored on a TimeService demo environment.
    """

    if not NOTE_FILE.exists():
        return {
            "exists": False,
            "text": "",
        }

    return {
        "exists": True,
        "text": NOTE_FILE.read_text(encoding="utf-8"),
    }


@mcp.tool()
def write_note(text: str) -> dict:
    """
    Write text to the demo note on a TimeService demo environment.

    Args:
        text: Text that should be stored in the note.
    """

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    NOTE_FILE.write_text(
        text,
        encoding="utf-8",
    )

    return {
        "success": True,
        "characters_written": len(text),
    }


@mcp.tool()
def check_note(expected_text: str) -> dict:
    """
    Check whether the TimeService demo note contains
    exactly the expected text.

    Args:
        expected_text: Text expected to be stored in the note.
    """

    if not NOTE_FILE.exists():
        return {
            "matches": False,
            "reason": "Note does not exist.",
        }

    actual_text = NOTE_FILE.read_text(
        encoding="utf-8"
    )

    return {
        "matches": actual_text == expected_text,
        "actual_text": actual_text,
    }


# ============================================================
# TIME SERVICE TOOLS
# ============================================================

@mcp.tool()
def time_service_start() -> dict:
    """
    Start the demo time web service on a TimeService demo environment.

    The service must be implemented in:
    ~/mcp-demo/time_server.py

    The service is expected to listen on port 8080.
    """

    if not TIME_SERVICE_FILE.exists():
        return {
            "success": False,
            "error": (
                "TimeService demo time_server.py does not exist. "
                "Create it before starting the service."
            ),
        }

    # Check whether it already appears to be running
    status = _check_time_service()

    if status["running"]:
        return {
            "success": True,
            "already_running": True,
            "url": status["url"],
        }

    DEMO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log = open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    )

    try:
        process = subprocess.Popen(
            [
                "python3",
                str(TIME_SERVICE_FILE),
            ],
            cwd=str(DEMO_DIR),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    except Exception as e:
        log.close()

        return {
            "success": False,
            "error": str(e),
        }

    PID_FILE.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    log.close()

    return {
        "success": True,
        "pid": process.pid,
        "url": f"http://brainbox:{TIME_SERVICE_PORT}",
    }


@mcp.tool()
def time_service_status() -> dict:
    """
    Check whether the demo time web service is responding.

    Performs an HTTP request to the service on port 8080.
    """

    return _check_time_service()


# ============================================================
# Internal helpers
# These are NOT exposed as MCP tools.
# ============================================================

def _check_time_service() -> dict:

    url = f"http://127.0.0.1:{TIME_SERVICE_PORT}"

    try:
        with urllib.request.urlopen(
            url,
            timeout=2,
        ) as response:

            body = response.read(
                500
            ).decode(
                "utf-8",
                errors="replace",
            )

            return {
                "running": True,
                "http_status": response.status,
                "url": f"http://brainbox:{TIME_SERVICE_PORT}",
                "response_preview": body,
            }

    except Exception as e:

        return {
            "running": False,
            "url": f"http://brainbox:{TIME_SERVICE_PORT}",
            "error": str(e),
        }


# ============================================================
# Run MCP server
# ============================================================

if __name__ == "__main__":

    DEMO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        streamable_http_path="/mcp",
    )
    