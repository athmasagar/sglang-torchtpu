#!/usr/bin/env python3
"""Set each CI ClusterQueue's nominalQuota from live allocatable TPU chips.

Run by _tpu-test.yml right before it submits a JobSet, so quota reflects the cluster at
the moment admission is decided. Between submissions a stale quota affects nothing --
nothing is consuming it -- and waitForPodsReady covers capacity lost while a job is
already queued or running.

A flavor is skipped unless it selects on ci-tpu-pool, so non-CI queues are untouched.
cpu and memory are derived from the chip count (pods take 4 chips / 4 cpu / 32Gi, so
1 cpu and 8Gi per chip) to keep TPU the only binding resource.

Any error fails the job: a stale quota admits jobs onto chips that do not exist, or
leaves live chips idle, and nothing downstream reports either.

Usage: reconcile_quota.py [--dry-run] [--pool=<ci-tpu-pool>]
"""

import json
import os
import re
import subprocess
import sys

import yaml

POOL_LABEL = "ci-tpu-pool"
TPU = "google.com/tpu"
CHIPS_PER_HOST = 4  # tpu7x-standard-4t
TOPOLOGY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "topology.yaml"
)


def kget(kind):
    out = subprocess.run(
        ["kubectl", "get", kind, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    return json.loads(out)["items"]


def ready(node):
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in node["status"].get("conditions", [])
    )


def nodes_per_slice(node):
    """How many nodes one slice of this node's topology spans.

    2x2x1 = 4 chips = 1 node (each node its own slice); 2x2x2 = 8 = 2; 2x2x4 = 16 = 4.
    """
    labels = node["metadata"]["labels"]
    dims = labels.get("cloud.google.com/gke-tpu-topology", "")
    if not re.fullmatch(r"\d+(x\d+)*", dims):
        return 1
    chips = 1
    for d in dims.split("x"):
        chips *= int(d)
    per_node = int(labels.get("cloud.google.com/gke-accelerator-count", 4)) or 4
    return max(1, chips // per_node)


def check_single_slice(nodes, pool):
    """A multi-host pool must hold exactly one slice.

    GKE tpu7x nodes carry no per-slice label -- ci-tpu-pool is the finest selector there
    is -- so two slices behind one label lets a JobSet's pods straddle them. They start,
    find no ICI path to each other, and hang until the deadline with no error saying why.
    Resizing a multi-host pool is the way this happens.
    """
    members = [
        n for n in nodes if n["metadata"].get("labels", {}).get(POOL_LABEL) == pool
    ]
    if not members:
        return
    want = nodes_per_slice(members[0])
    if want > 1 and len(members) != want:
        topo = members[0]["metadata"]["labels"].get("cloud.google.com/gke-tpu-topology")
        sys.exit(
            f"pool {pool!r} has {len(members)} nodes but one {topo} slice is {want}. "
            "Pods could straddle two slices and hang at ICI rendezvous -- add capacity as "
            "a NEW pool with its own ci-tpu-pool label, never by resizing this one."
        )


def chips_per_pool(nodes):
    chips = {}
    for n in nodes:
        pool = n["metadata"].get("labels", {}).get(POOL_LABEL)
        if pool and ready(n):
            alloc = int(n["status"]["allocatable"].get(TPU, 0))
            chips[pool] = chips.get(pool, 0) + alloc
    return chips


def _quantity(n, unit):
    # The API server stores a zero quantity as "0", whatever unit it was written with.
    return f"{n}{unit}" if n else "0"


def _resources(chips, lend_tpu=True):
    """TPU quota plus the cpu/memory derived from it, in the ClusterQueue's own shape."""
    out = []
    for name, per_chip, unit in ((TPU, 1, ""), ("cpu", 1, ""), ("memory", 8, "Gi")):
        r = {"name": name, "nominalQuota": _quantity(n=chips * per_chip, unit=unit)}
        if name == TPU and not lend_tpu:
            r["lendingLimit"] = "0"
        out.append(r)
    return out


def _shared_pool_quotas(topo, chips):
    """{(queue, pool): (chips, lend_tpu)} for pools that two lanes list in topology.yaml.

    The lane with `min_hosts` gets that many hosts (or every live one, if fewer) and never
    lends them; the other lane gets the rest and lends what it is not using.
    """
    users = {}
    for slices in topo["lanes"].values():
        for entry in slices.values():
            for pool in entry["pools"]:
                users.setdefault(pool, []).append(entry)
    out = {}
    for pool, entries in users.items():
        if len(entries) < 2:
            continue
        keepers = [e for e in entries if "min_hosts" in e]
        if len(entries) != 2 or len(keepers) != 1:
            sys.exit(
                f"shared pool {pool!r} needs two lanes, exactly one with min_hosts"
            )
        keeper = keepers[0]
        other = next(e for e in entries if e is not keeper)
        live = chips.get(pool, 0)
        kept = min(live, keeper["min_hosts"] * CHIPS_PER_HOST)
        out[(keeper["queue"], pool)] = (kept, False)
        out[(other["queue"], pool)] = (live - kept, True)
    return out


def main():
    dry_run = "--dry-run" in sys.argv
    pool = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pool=")), "")

    nodes = kget("nodes")
    if not nodes:
        sys.exit("no nodes returned -- refusing to zero every quota")
    if pool:
        check_single_slice(nodes, pool)
    chips = chips_per_pool(nodes)

    flavor_pool = {
        f["metadata"]["name"]: f["spec"]["nodeLabels"][POOL_LABEL]
        for f in kget("resourceflavors")
        if (f.get("spec") or {}).get("nodeLabels", {}).get(POOL_LABEL)
    }
    cqs = kget("clusterqueues")
    with open(TOPOLOGY_FILE) as f:
        shared = _shared_pool_quotas(topo=yaml.safe_load(f), chips=chips)

    for cq in cqs:
        name = cq["metadata"]["name"]
        groups, changed = [], False
        for g in cq["spec"].get("resourceGroups", []):
            flavors = []
            for fl in g.get("flavors", []):
                pool = flavor_pool.get(fl["name"])
                if pool is None:
                    flavors.append(fl)
                    continue
                n, lend = shared.get((name, pool), (chips.get(pool, 0), True))
                want = _resources(chips=n, lend_tpu=lend)
                have = [{k: str(v) for k, v in r.items()} for r in fl["resources"]]
                if have != want:
                    changed = True
                    print(f"{name}/{fl['name']} ({pool}): {have} -> {want}")
                flavors.append({"name": fl["name"], "resources": want})
            groups.append(
                {"coveredResources": g["coveredResources"], "flavors": flavors}
            )
        if not changed:
            continue
        if dry_run:
            print(f"(dry-run) would patch {name}")
            continue
        subprocess.run(
            [
                "kubectl",
                "patch",
                "clusterqueue",
                name,
                "--type=merge",
                "-p",
                json.dumps({"spec": {"resourceGroups": groups}}),
            ],
            check=True,
            timeout=60,
        )
        print(f"patched {name}")


if __name__ == "__main__":
    main()
