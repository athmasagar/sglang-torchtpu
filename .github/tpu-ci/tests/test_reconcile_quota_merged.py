"""The merged 2x2x1 pool: pr-gate keeps its hosts unlent, nightly gets the rest."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reconcile_quota

TOPO = {
    "lanes": {
        "pr-gate": {1: {"pools": ["ci-2x2x1"], "queue": "tpu-prgate-1h"}},
        "nightly": {
            1: {"pools": ["ci-2x2x1"], "queue": "tpu-nightly-1h"},
            2: {"pools": ["nightly"], "queue": "tpu-nightly-2h"},
        },
    },
    "merged_pools": {"ci-2x2x1": {"pr_gate_hosts": 5}},
}


def split(hosts):
    return reconcile_quota.merged_quotas(topo=TOPO, chips={"ci-2x2x1": 4 * hosts})


class TestMergedQuotas(unittest.TestCase):
    def test_pr_gate_keeps_its_hosts_and_nightly_gets_the_rest(self):
        got = split(hosts=8)
        self.assertEqual(got[("tpu-prgate-1h", "ci-2x2x1")], (20, False))
        self.assertEqual(got[("tpu-nightly-1h", "ci-2x2x1")], (12, True))

    def test_quotas_sum_to_live_chips(self):
        # Over-quota admits jobs that cannot schedule; under-quota idles live hosts.
        for hosts in range(0, 12):
            got = split(hosts=hosts)
            self.assertEqual(sum(n for n, _ in got.values()), 4 * hosts, hosts)

    def test_fewer_hosts_than_the_floor_all_go_to_pr_gate(self):
        got = split(hosts=3)
        self.assertEqual(got[("tpu-prgate-1h", "ci-2x2x1")], (12, False))
        self.assertEqual(got[("tpu-nightly-1h", "ci-2x2x1")], (0, True))

    def test_unmerged_pools_are_left_to_the_default(self):
        self.assertNotIn(("tpu-nightly-2h", "nightly"), split(hosts=8))

    def test_only_tpu_carries_the_lending_limit(self):
        self.assertEqual(
            reconcile_quota.resources(chips=8, lend_tpu=False),
            [
                {"name": "google.com/tpu", "nominalQuota": "8", "lendingLimit": "0"},
                {"name": "cpu", "nominalQuota": "8"},
                {"name": "memory", "nominalQuota": "64Gi"},
            ],
        )
        self.assertNotIn("lendingLimit", reconcile_quota.resources(chips=4)[0])


if __name__ == "__main__":
    unittest.main()
