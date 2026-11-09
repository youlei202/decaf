#!/usr/bin/env python3
"""Record actual local/SSH GPU capacity without starting or stopping workloads."""

from __future__ import annotations

import argparse
import csv
import io
import json
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def inspect(target: tuple[str, str | None, int | None], identity: Path | None) -> dict:
    name, host, port = target
    command = QUERY
    if host:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-p", str(port)]
        if identity:
            command.extend(["-o", "IdentitiesOnly=yes", "-i", str(identity)])
        command.extend([host, " ".join(QUERY)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=True)
    except (subprocess.SubprocessError, OSError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return {"name": name, "host": host, "port": port, "error": detail.strip()}
    gpus = []
    for row in csv.reader(io.StringIO(result.stdout)):
        index, uuid, model, total, used, utilization = [value.strip() for value in row]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "model": model,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
                "idle_at_probe": int(used) < 1024 and int(utilization) < 10,
            }
        )
    return {"name": name, "host": host, "port": port, "gpus": gpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--host", default="leiyo@ssh.cloud.sdu.dk")
    parser.add_argument("--ports", type=int, nargs="+", default=[2813, 2550, 2702, 2064])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = [(socket.gethostname(), None, None)]
    targets.extend((f"ssh-{port}", args.host, port) for port in args.ports)
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        nodes = list(pool.map(lambda target: inspect(target, args.identity), targets))
    unique = {gpu["uuid"]: gpu for node in nodes for gpu in node.get("gpus", [])}
    report = {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "nodes": nodes,
        "unique_gpu_count": len(unique),
        "idle_gpu_count": sum(gpu["idle_at_probe"] for gpu in unique.values()),
        "note": "An observation, not a reservation; existing jobs must not be interrupted.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
