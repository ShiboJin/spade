"""Memory safety checks without allocating large tensors or requiring Docker."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import memory_guard as guard


class MemoryGuardTests(unittest.TestCase):
    def test_gpu_locks_across_processes_and_release_on_exit(self):
        # A real separate launcher holds GPUs, with only Docker/RAM probing mocked.
        code = '''
import sys
from pathlib import Path
from unittest.mock import patch
from scripts.memory_guard import task_resources
cfg = dict(gpu_ids=[0, 1, 2, 3], memory_limit_gib=80, host_memory_reserve_gib=48)
with patch("scripts.memory_guard.concurrent_preflight"), task_resources(cfg, Path(sys.argv[1])):
    print("ready", flush=True)
    sys.stdin.read()
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = subprocess.Popen([sys.executable, "-c", code, directory],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                import select
                self.assertTrue(select.select([process.stdout], [], [], 10)[0], "launcher did not become ready")
                self.assertEqual(process.stdout.readline().strip(), "ready")
                cfg = dict(gpu_ids=[4, 5, 6, 7], memory_limit_gib=80, host_memory_reserve_gib=48)
                with patch.object(guard, "concurrent_preflight") as check:
                    with guard.task_resources(cfg, root):
                        self.assertEqual(check.call_args.args[2], 80)
                    with self.assertRaisesRegex(RuntimeError, "GPU 3"):
                        with guard.task_resources({**cfg, "gpu_ids": [3, 4]}, root):
                            self.fail("overlapping GPUs admitted")
                process.kill()
                process.wait(timeout=10)
                with patch.object(guard, "concurrent_preflight") as check:
                    with guard.task_resources({**cfg, "gpu_ids": [0, 1, 2, 3]}, root):
                        self.assertEqual(check.call_args.args[2], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=10)

    def test_failed_admission_releases_gpu_reservations(self):
        cfg = dict(gpu_ids=[0, 1], memory_limit_gib=80, host_memory_reserve_gib=48)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(guard, "concurrent_preflight", side_effect=RuntimeError("RAM")):
                with self.assertRaisesRegex(RuntimeError, "RAM"):
                    with guard.task_resources(cfg, root):
                        self.fail("admitted without RAM")
            with patch.object(guard, "concurrent_preflight") as check:
                with guard.task_resources(cfg, root):
                    self.assertEqual(check.call_args.args[2], 0)

    def test_preflight_refuses_unsafe_host(self):
        info = dict(MemoryLimit=True, SwapLimit=False, CgroupVersion="1", MemTotal=252 * guard.GIB)
        for changes, available, message in (
            ({"MemoryLimit": False}, 240, "cannot enforce"),
            ({"CgroupVersion": "2"}, 240, "no-swap"),
            ({}, 180, "Not enough"),
        ):
            with self.subTest(changes=changes, available=available), patch.object(
                guard.subprocess, "check_output", return_value=json.dumps({**info, **changes})
            ), patch.object(guard, "memory_available", return_value=available * guard.GIB):
                with self.assertRaisesRegex(RuntimeError, message):
                    guard.preflight(160, 48)

    def test_v1_without_swap_accounting_and_duplicate_launch(self):
        info = dict(MemoryLimit=True, SwapLimit=False, CgroupVersion="1", MemTotal=252 * guard.GIB)
        for active in ("", "spade-train-existing"):
            with patch.object(guard.subprocess, "check_output", side_effect=[json.dumps(info), active]), patch.object(
                guard, "memory_available", return_value=240 * guard.GIB
            ):
                if active:
                    with self.assertRaisesRegex(RuntimeError, "already running"):
                        guard.preflight(160, 48)
                else:
                    guard.preflight(160, 48)

    def test_real_cgroup_values_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            memory.mkdir()
            (memory / "memory.limit_in_bytes").write_text(str(160 * guard.GIB))
            (memory / "memory.swappiness").write_text("0")
            (memory / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\n")
            guard.verify_container_limits(160, root)
            for filename, value in (("memory.swappiness", "60"),
                                    ("memory.limit_in_bytes", str(161 * guard.GIB)),
                                    ("memory.oom_control", "oom_kill_disable 1\n")):
                path = memory / filename
                old = path.read_text()
                path.write_text(value)
                with self.assertRaisesRegex(RuntimeError, "protection is missing"):
                    guard.verify_container_limits(160, root)
                path.write_text(old)

    def test_cgroup_v2_no_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.max").write_text(str(96 * guard.GIB))
            (root / "memory.swap.max").write_text("0")
            guard.verify_container_limits(96, root)
            (root / "memory.max").write_text("max")
            with self.assertRaises(RuntimeError):
                guard.verify_container_limits(96, root)

    def test_low_memory_kills_only_owned_container(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("docker", 2), 137]
        with patch.object(guard, "memory_available", return_value=20 * guard.GIB), patch.object(
            guard.subprocess, "run"
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "Memory guard stopped"):
                guard.guarded_wait(process, "spade-train-owned", 48)
        self.assertEqual(run.call_args.args[0], ["docker", "kill", "spade-train-owned"])

    def test_success_does_not_kill_container(self):
        process = Mock()
        process.wait.return_value = 0
        with patch.object(guard.subprocess, "run") as run:
            self.assertEqual(guard.guarded_wait(process, "test", 48), 0)
        run.assert_not_called()

    def test_concurrent_launcher_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launch.lock"
            with guard.launch_lock(path):
                with self.assertRaisesRegex(RuntimeError, "launcher is active"):
                    with guard.launch_lock(path):
                        self.fail("second launcher acquired lock")
            with guard.launch_lock(path):
                pass

    def test_invalid_limits(self):
        for limit, reserve in ((0, 48), (True, 48), (160, 0), (160, 31)):
            with self.assertRaises(ValueError):
                guard.validate_limits(limit, reserve)
