import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('telemetry', Path(__file__).resolve().parents[1]/'linux-cgroup-telemetry.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class CgroupTelemetryTests(unittest.TestCase):
    def capture(self, membership, mountinfo, files):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root/'proc/71'
            proc.mkdir(parents=True)
            (proc/'cgroup').write_text(membership)
            (proc/'mountinfo').write_text(mountinfo)
            for name, value in files.items():
                path = root/'fs'/name.lstrip('/')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            return subject.capture(71, root/'proc', root/'fs')

    def test_hybrid_namespaced_v1_and_combined_cpu_mount(self):
        row = self.capture('12:memory:/docker/id\n5:cpu,cpuacct:/docker/id\n0::/docker/id\n',
            '1 0 0:1 /docker/id /cg/mem rw - cgroup cgroup rw,memory\n'
            '2 0 0:2 /docker/id /cg/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n'
            '3 0 0:3 /docker/id /cg/v2 rw - cgroup2 cgroup rw\n',
            {'/cg/mem/memory.usage_in_bytes':'120', '/cg/mem/memory.limit_in_bytes':'1000',
             '/cg/mem/memory.max_usage_in_bytes':'200', '/cg/mem/memory.failcnt':'3',
             '/cg/mem/memory.use_hierarchy':'1', '/cg/cpu,cpuacct/cpu.stat':'throttled_time 77',
             '/cg/cpu,cpuacct/cpu.cfs_quota_us':'1615000', '/cg/cpu,cpuacct/cpu.cfs_period_us':'100000',
             '/cg/v2/memory.current':'9999'})
        self.assertEqual(row['memory.current'], '120')
        self.assertEqual(row['memory.max'], '1000')
        self.assertEqual(row['memory.peak'], '200')
        self.assertEqual(row['memory.failcnt'], '3')
        self.assertNotIn('memory.events', row)
        self.assertEqual(row['cpu.cfs_quota_us'], '1615000')
        self.assertEqual(row['controllers']['memory']['version'], 1)
        self.assertFalse(row['controllers']['memory']['ancestors_above_mount_visible'])
        self.assertFalse(row['errors'])

    def test_v2_target_child_and_visible_ancestors(self):
        row = self.capture('0::/team/job\n',
            '1 0 0:1 / /cg rw - cgroup2 cgroup rw\n'
            '2 0 0:1 /team/job /bind rw - cgroup2 cgroup rw\n',
            {'/cg/team/job/memory.current':'11', '/cg/team/job/memory.max':'max',
             '/cg/team/job/memory.events':'oom 0', '/cg/team/job/cpu.stat':'throttled_usec 5',
             '/cg/team/job/cpu.max':'max 100000', '/cg/team/memory.current':'30',
             '/cg/team/memory.max':'100'})
        self.assertEqual(row['memory.current'], '11')
        self.assertEqual(row['memory.max'], 'max')
        self.assertEqual(row['cpu.stat'], 'throttled_usec 5')
        ancestors = row['controllers']['memory']['visible_ancestors']
        self.assertEqual([a['path'] for a in ancestors], ['/cg/team/job', '/cg/team', '/cg'])
        self.assertEqual(ancestors[1]['limit'], '100')
        self.assertEqual(row['controllers']['memory']['version'], 2)

    def test_escaped_mountpoint_and_nonhierarchical_v1_parent(self):
        row = self.capture('1:memory:/job\n', '1 0 0:1 / /cg\\040dir rw - cgroup cgroup rw,memory\n',
            {'/cg dir/job/memory.usage_in_bytes':'1', '/cg dir/job/memory.limit_in_bytes':'20',
             '/cg dir/job/memory.use_hierarchy':'1', '/cg dir/memory.limit_in_bytes':'2',
             '/cg dir/memory.usage_in_bytes':'2', '/cg dir/memory.use_hierarchy':'0'})
        self.assertEqual(row['controllers']['memory']['path'], '/cg dir/job')
        self.assertEqual(row['controllers']['memory']['visible_ancestors'][1]['use_hierarchy'], '0')

    def test_unreadable_or_unmapped_usage_stays_unknown(self):
        row = self.capture('0::/job\n', '1 0 0:1 / /cg rw - cgroup2 cgroup rw\n', {})
        self.assertNotIn('memory.current', row)
        self.assertTrue(row['errors'])
        row = self.capture('1:memory:/unmapped\n', '1 0 0:1 /other /cg rw - cgroup cgroup rw,memory\n', {})
        self.assertTrue(row['errors'])

    def test_traversal_and_duplicate_membership_are_rejected(self):
        for membership in ('0::/a/../b\n', '0::/a\n0::/b\n'):
            with self.assertRaises(ValueError):
                subject.controller_mounts(membership, '')


if __name__ == '__main__':
    unittest.main()
