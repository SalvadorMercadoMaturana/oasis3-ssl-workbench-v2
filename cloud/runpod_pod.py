#!/usr/bin/env python
"""Provision, inspect and terminate the RunPod GPU pod (runs on the local machine).

    python cloud/runpod_pod.py create [--max-price 0.8] [--pubkey ~/.ssh/runpod_ed25519.pub]
    python cloud/runpod_pod.py wait          # until SSH is reachable; prints the ssh command
    python cloud/runpod_pod.py status | ssh | balance | cost
    python cloud/runpod_pod.py terminate     # deletes the pod (billing stops)

The API key comes from ``RUNPOD_API_KEY`` or the gitignored ``.env``. ``create``
walks a preference list — RTX 4090 community, RTX 4090 secure, then other
24 GB+ GPUs — skipping every option whose current on-demand price exceeds
``--max-price`` (USD/h) and every option without capacity, and asks for the
most vCPUs first (CPU augmentation feeds three concurrent trainings). The pod
state (id, price, timestamps) is kept in ``cloud/state/pod.json`` (gitignored)
so ``cost`` can report the real spend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "cloud" / "state" / "pod.json"
API = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# (gpu type id, cloud) in order of preference; prices are checked live. The RTX
# 5090 community pods (0.69 USD/h, 32 GB, ~20 vCPU) beat the 4090 for these
# CPU-augmentation-bound runs, so they come first.
PREFERENCE = [
    ("NVIDIA GeForce RTX 5090", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA RTX 6000 Ada Generation", "COMMUNITY"),
    ("NVIDIA L40", "COMMUNITY"),
    ("NVIDIA GeForce RTX 3090", "COMMUNITY"),
    ("NVIDIA RTX A6000", "COMMUNITY"),
    ("NVIDIA GeForce RTX 3090", "SECURE"),
]
VCPU_STEPS = (16, 12, 8)


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    env = ROOT / ".env"
    if not key and env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("RUNPOD_API_KEY missing (environment or .env)")
    return key


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key()}",
        "User-Agent": "oasis3-ssl-workbench"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            doc = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()[:500]}") from exc
    if doc.get("errors"):
        raise RuntimeError("; ".join(e.get("message", str(e)) for e in doc["errors"]))
    return doc["data"]


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def prices() -> dict[str, dict[str, float | None]]:
    data = gql("query { gpuTypes { id securePrice communityPrice memoryInGb } }")
    return {g["id"]: {"SECURE": g["securePrice"], "COMMUNITY": g["communityPrice"],
                      "memory": g["memoryInGb"]} for g in data["gpuTypes"]}


def create(args: argparse.Namespace) -> None:
    state = load_state()
    if state.get("id") and not state.get("terminated_at"):
        sys.exit(f"pod {state['id']} is still recorded as live; terminate it first")
    pubkey = Path(os.path.expanduser(args.pubkey)).read_text(encoding="utf-8").strip()
    table = prices()
    mutation = """mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) {
        id costPerHr machineId machine { gpuDisplayName } } }"""
    tried = []
    preference = [(g, c) for g, c in PREFERENCE
                  if not args.only or any(o.lower() in g.lower() for o in args.only.split(","))]
    for gpu, cloud in preference:
        price = (table.get(gpu) or {}).get(cloud)
        if not price or price > args.max_price:
            tried.append(f"{gpu}/{cloud}: price {price} > {args.max_price} or unlisted")
            continue
        for vcpu in VCPU_STEPS:
            payload = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu,
                       "name": args.name, "imageName": args.image,
                       "containerDiskInGb": args.disk, "volumeInGb": args.volume,
                       "volumeMountPath": "/workspace", "ports": "22/tcp,8888/http",
                       "minVcpuCount": vcpu, "minMemoryInGb": args.min_ram,
                       "supportPublicIp": True, "startSsh": True,
                       "allowedCudaVersions": ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9",
                                               "13.0", "13.1", "13.2"],
                       "env": [{"key": "PUBLIC_KEY", "value": pubkey}]}
            try:
                pod = gql(mutation, {"input": payload})["podFindAndDeployOnDemand"]
            except RuntimeError as exc:
                tried.append(f"{gpu}/{cloud}/{vcpu}vCPU: {str(exc)[:120]}")
                continue
            if not pod:
                tried.append(f"{gpu}/{cloud}/{vcpu}vCPU: no pod returned")
                continue
            if float(pod["costPerHr"]) > args.max_price:  # never keep an over-budget pod
                terminate_pod(pod["id"])
                tried.append(f"{gpu}/{cloud}: billed {pod['costPerHr']} > {args.max_price}")
                break
            save_state({"id": pod["id"], "gpu": gpu, "cloud": cloud, "min_vcpu": vcpu,
                        "cost_per_hr": float(pod["costPerHr"]), "created_at": now(),
                        "image": args.image, "machine": pod.get("machine")})
            print(json.dumps(load_state(), indent=2))
            return
    print("\n".join(tried))
    sys.exit("no GPU option available within budget")


def pod_info(pod_id: str) -> dict:
    query = """query { pod(input: {podId: "POD_ID"}) { id name desiredStatus costPerHr
      runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type }
                gpus { id gpuUtilPercent memoryUtilPercent } }
      machine { gpuDisplayName } } }""".replace("POD_ID", pod_id)
    return gql(query)["pod"]


def ssh_target(info: dict) -> tuple[str, int] | None:
    for port in ((info or {}).get("runtime") or {}).get("ports") or []:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], int(port["publicPort"])
    return None


def wait(args: argparse.Namespace) -> None:
    state = load_state()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        info = pod_info(state["id"])
        target = ssh_target(info)
        if target:
            state.update(ssh_host=target[0], ssh_port=target[1], ready_at=now())
            save_state(state)
            print(f"ssh -i ~/.ssh/runpod_ed25519 -p {target[1]} root@{target[0]}")
            return
        print(f"waiting: {info.get('desiredStatus') if info else 'unknown'}", flush=True)
        time.sleep(10)
    sys.exit("timed out waiting for SSH")


def terminate_pod(pod_id: str) -> None:
    gql('mutation { podTerminate(input: {podId: "POD_ID"}) }'.replace("POD_ID", pod_id))


def terminate(args: argparse.Namespace) -> None:
    del args
    state = load_state()
    if not state.get("id"):
        sys.exit("no pod recorded")
    terminate_pod(state["id"])
    state["terminated_at"] = now()
    save_state(state)
    cost(argparse.Namespace())


def cost(args: argparse.Namespace) -> None:
    del args
    state = load_state()
    start = datetime.fromisoformat(state["created_at"])
    end = datetime.fromisoformat(state["terminated_at"]) if state.get("terminated_at") \
        else datetime.now(UTC)
    hours = (end - start).total_seconds() / 3600
    me = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
    print(json.dumps({"pod": state["id"], "gpu": state["gpu"], "cloud": state["cloud"],
                      "cost_per_hr": state["cost_per_hr"], "hours": round(hours, 3),
                      "estimated_cost_usd": round(hours * state["cost_per_hr"], 3),
                      "balance_usd": me["clientBalance"],
                      "current_spend_per_hr": me["currentSpendPerHr"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("create")
    c.add_argument("--max-price", type=float, default=0.8)
    c.add_argument("--pubkey", default="~/.ssh/runpod_ed25519.pub")
    c.add_argument("--name", default="oasis3-ssl-4090")
    c.add_argument("--image", default=IMAGE)
    c.add_argument("--disk", type=int, default=40, help="container disk (GB)")
    c.add_argument("--volume", type=int, default=20, help="/workspace volume (GB)")
    c.add_argument("--min-ram", type=int, default=30)
    c.add_argument("--only", default=None, help="comma list of GPU name fragments, e.g. 5090,4090")
    w = sub.add_parser("wait")
    w.add_argument("--timeout", type=int, default=900)
    sub.add_parser("status")
    sub.add_parser("ssh")
    sub.add_parser("balance")
    sub.add_parser("cost")
    sub.add_parser("terminate")
    args = parser.parse_args()
    if args.command == "create":
        create(args)
    elif args.command == "wait":
        wait(args)
    elif args.command == "status":
        print(json.dumps(pod_info(load_state()["id"]), indent=2))
    elif args.command == "ssh":
        state = load_state()
        print(f"ssh -i ~/.ssh/runpod_ed25519 -p {state['ssh_port']} root@{state['ssh_host']}")
    elif args.command == "balance":
        print(json.dumps(gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]))
    elif args.command == "cost":
        cost(args)
    else:
        terminate(args)


if __name__ == "__main__":
    main()
