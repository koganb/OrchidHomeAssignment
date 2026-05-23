from __future__ import annotations

import subprocess
import sys
import time
import os
import signal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
API_COMMAND = [
    sys.executable,
    "-m",
    "uvicorn",
    "api:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]
STREAMLIT_COMMAND = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "streamlit_gui/app.py",
]


def main() -> int:
    processes: list[subprocess.Popen] = []

    try:
        processes.append(_start("FastAPI", API_COMMAND))
        processes.append(_start("Streamlit", STREAMLIT_COMMAND))

        while True:
            for process in processes:
                if process.poll() is not None:
                    return process.returncode or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        _stop_all(processes)


def _start(name: str, command: list[str]) -> subprocess.Popen:
    print(f"Starting {name}: {' '.join(command)}", flush=True)
    kwargs = {"cwd": ROOT}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def _stop_all(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            _terminate(process)

    deadline = time.monotonic() + 10
    for process in processes:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)

    for process in processes:
        if process.poll() is None:
            process.kill()


def _terminate(process: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)


if __name__ == "__main__":
    raise SystemExit(main())
