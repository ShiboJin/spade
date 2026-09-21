"""Host and cgroup memory safety for the Docker training/evaluation launchers."""
from contextlib import contextmanager, ExitStack
import fcntl
import json
from pathlib import Path
import subprocess

GIB = 1024 ** 3
DEFAULT_RESERVE_GIB = 48


def validate_limits(limit_gib, reserve_gib):
    for name, value in (("memory_limit_gib", limit_gib), ("host_memory_reserve_gib", reserve_gib)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if reserve_gib < 32:
        raise ValueError("host_memory_reserve_gib must be at least 32")


def memory_available():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0]) * 1024


def docker_memory_args(limit_gib):
    return ["--memory", f"{limit_gib}g", "--memory-swap", f"{limit_gib}g",
            "--memory-swappiness", "0", "--label", "spade.memory-guard=true"]


def preflight(limit_gib, reserve_gib):
    validate_limits(limit_gib, reserve_gib)
    info = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True, timeout=15))
    if not info.get("MemoryLimit"):
        raise RuntimeError("Docker cannot enforce a memory limit; refusing to start")
    if not info.get("SwapLimit") and str(info.get("CgroupVersion")) != "1":
        raise RuntimeError("Docker cannot enforce no-swap operation; refusing to start")
    available = memory_available()
    if min(available, info["MemTotal"]) < (limit_gib + reserve_gib) * GIB:
        raise RuntimeError(
            f"Not enough host RAM: {available / GIB:.1f} GiB available; need "
            f"{limit_gib} GiB task budget + {reserve_gib} GiB host reserve. No task started."
        )
    active = subprocess.check_output(
        ["docker", "ps", "--filter", "label=spade.memory-guard=true", "--format", "{{.Names}}"],
        text=True, timeout=15,
    ).strip()
    if active:
        raise RuntimeError(f"A guarded task is already running: {active}. Wait for it to finish.")



@contextmanager
def task_resources(cfg, root):
    """Reserve GPUs and RAM across training and evaluation launchers."""
    validate_limits(cfg["memory_limit_gib"], cfg["host_memory_reserve_gib"])
    # Keep historical lock paths so already-running evals remain protected.
    with ExitStack() as stack:
        gate = stack.enter_context((root / ".spade-memory.lock").open("a"))
        try:
            fcntl.flock(gate, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A legacy training/evaluation launcher is active; wait for it to finish") from exc
        # Serialize admission, including launchers which have not started Docker yet.
        with (root / ".spade-eval-admission.lock").open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            reserved = 0.0
            capacity = memory_available() / GIB
            capacities = []
            reserve = cfg["host_memory_reserve_gib"]
            for path in root.glob(".spade-eval-gpu-*.lock"):
                with path.open("r+") as existing:
                    try:
                        fcntl.flock(existing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        record = json.load(existing)
                        reserved += record["budget_gib"]
                        reserve = max(reserve, record["reserve_gib"])
                        capacities.append(record["capacity_gib"])
            capacity = min([capacity + reserved, *capacities])
            try:
                gpu_locks = []
                for gpu in sorted(cfg["gpu_ids"]):
                    lock = stack.enter_context((root / f".spade-eval-gpu-{gpu}.lock").open("a+"))
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise RuntimeError(f"GPU {gpu} is already reserved by another training/evaluation task") from exc
                    gpu_locks.append(lock)
                concurrent_preflight(cfg["memory_limit_gib"], reserve, reserved, capacity)
                for lock in gpu_locks:
                    lock.seek(0)
                    lock.truncate()
                    json.dump(dict(budget_gib=cfg["memory_limit_gib"] / len(gpu_locks),
                                   capacity_gib=capacity,
                                   reserve_gib=cfg["host_memory_reserve_gib"]), lock)
                    lock.flush()
            except BaseException:
                # Release partial reservations before another admission can inspect them.
                stack.close()
                raise
        yield


def concurrent_preflight(limit_gib, reserve_gib, reserved_gib, capacity_gib):
    info = json.loads(subprocess.check_output(
        ["docker", "info", "--format", "{{json .}}"], text=True, timeout=15))
    if not info.get("MemoryLimit") or (not info.get("SwapLimit") and str(info.get("CgroupVersion")) != "1"):
        raise RuntimeError("Docker cannot enforce RAM/no-swap limits; refusing to start")
    available = min(memory_available(), info["MemTotal"]) / GIB
    # Reserve aggregate budgets against admission capacity, and also check live RAM.
    if (min(capacity_gib, info["MemTotal"] / GIB) < limit_gib + reserved_gib + reserve_gib
            or available < limit_gib + reserve_gib):
        raise RuntimeError(
            f"Not enough host RAM: {available:.1f} GiB available; need {limit_gib} GiB new task "
            f"+ {reserved_gib:g} GiB other task budgets + {reserve_gib} GiB host reserve")
    containers = subprocess.check_output(
        ["docker", "ps", "--filter", "label=spade.memory-guard=true",
         "--format", '{{.Names}}\t{{.Label "spade.eval-concurrent"}}\t{{.Label "spade.gpu-concurrent"}}'],
        text=True, timeout=15).strip()
    active = []
    for line in containers.splitlines():
        name, *labels = line.split("\t")
        if "true" not in labels:
            active.append(name)
    if active:
        raise RuntimeError(f"A legacy guarded container is running: {', '.join(active)}")


def verify_container_limits(limit_gib, root=Path("/sys/fs/cgroup")):
    """Fail before importing model libraries if the kernel did not apply limits."""
    v1 = root / "memory"
    if (v1 / "memory.limit_in_bytes").exists():
        limit = int((v1 / "memory.limit_in_bytes").read_text())
        no_swap = int((v1 / "memory.swappiness").read_text()) == 0
        oom_enabled = "oom_kill_disable 0" in (v1 / "memory.oom_control").read_text()
    else:
        raw = (root / "memory.max").read_text().strip()
        limit = int(raw) if raw != "max" else 0
        no_swap = (root / "memory.swap.max").read_text().strip() == "0"
        oom_enabled = True
    if not 0 < limit <= limit_gib * GIB or not no_swap or not oom_enabled:
        raise RuntimeError("Required cgroup RAM/no-swap/OOM protection is missing; refusing to load model")


@contextmanager
def launch_lock(path):
    # Held through task exit, so simultaneous launches cannot both pass preflight.
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another SPADE training/evaluation launcher is active") from exc
        yield


def guarded_wait(process, container_name, reserve_gib):
    try:
        while True:
            try:
                return process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                available = memory_available()
                if available < reserve_gib * GIB // 2:
                    raise RuntimeError(
                        f"Memory guard stopped {container_name}: only {available / GIB:.1f} GiB host RAM available"
                    )
    except BaseException:
        # Kill only this launcher's container, promptly freeing all worker memory.
        subprocess.run(["docker", "kill", container_name], check=False, timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait(timeout=20)
        raise


if __name__ == "__main__":
    import sys
    verify_container_limits(int(sys.argv[1]))
