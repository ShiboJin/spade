"""Memory safety checks without allocating large tensors or requiring Docker."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import memory_guard as guard


class MemoryGuardTests(unittest.TestCase):
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
