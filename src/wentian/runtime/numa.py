"""Checked NUMA/HBM placement for Wentian worker processes."""

import ctypes
import mmap
import os
import re
import sys

from wentian.settings import HBMSettings

_MIB = 1024.0 * 1024.0
_SYSCALL_NUMBERS = {
    "aarch64": {"set_mempolicy": 237, "move_pages": 239},
    "x86_64": {"set_mempolicy": 238, "move_pages": 279},
}


def _message(worker_id, message):
    sys.stderr.write(f"HBM worker={worker_id} {message}\n")
    sys.stderr.flush()


def _numa_residency_mb():
    page_size_re = re.compile(r"kernelpagesize_kB=(\d+)")
    node_re = re.compile(r"\bN(\d+)=(\d+)")
    residency = {}
    with open("/proc/self/numa_maps", errors="replace") as stream:
        for line in stream:
            match = page_size_re.search(line)
            page_size_kb = int(match.group(1)) if match else 4
            for node, count in node_re.findall(line):
                node = int(node)
                residency[node] = residency.get(node, 0.0) + int(count) * page_size_kb / 1024.0
    return residency


def _syscall_number(name):
    machine = os.uname().machine
    try:
        return _SYSCALL_NUMBERS[machine][name]
    except KeyError as exc:
        raise RuntimeError(f"{name} syscall is unknown on {machine}") from exc


def _page_residency_mb(address, length):
    page_size = mmap.PAGESIZE
    page_count = (length + page_size - 1) // page_size
    pages = (ctypes.c_void_p * page_count)(
        *(address + index * page_size for index in range(page_count))
    )
    status = (ctypes.c_int * page_count)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(
        ctypes.c_long(_syscall_number("move_pages")),
        ctypes.c_int(0),
        ctypes.c_ulong(page_count),
        pages,
        ctypes.c_void_p(),
        status,
        ctypes.c_int(0),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, "move_pages query failed")
    failures = [value for value in status if value < 0]
    if failures:
        raise OSError(-failures[0], "move_pages page query failed")

    page_mb = page_size / _MIB
    residency = {}
    for node in status:
        residency[node] = residency.get(node, 0.0) + page_mb
    return residency


def _create_probe(worker_id, hbm_node, probe_mb):
    probe_bytes = int(probe_mb * _MIB)
    if probe_bytes < mmap.PAGESIZE:
        raise ValueError("WT_HBM_PROBE_MB must allocate at least one page")

    probe = mmap.mmap(
        -1,
        probe_bytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    try:
        for offset in range(0, probe_bytes, mmap.PAGESIZE):
            probe[offset] = 0
        address = ctypes.addressof(ctypes.c_char.from_buffer(probe))
        actual_mb = probe_bytes / _MIB
        resident_mb = _page_residency_mb(address, probe_bytes).get(hbm_node, 0.0)
        if resident_mb < actual_mb * 0.9:
            raise RuntimeError(
                f"required HBM probe placed {resident_mb:.1f}/{actual_mb:.1f} MB on node {hbm_node}"
            )
    except BaseException:
        probe.close()
        raise

    _message(
        worker_id,
        f"probe node={hbm_node} resident_mb={resident_mb:.1f}/{actual_mb:.1f}",
    )
    return probe


def _node_capacity_kb(hbm_node):
    meminfo = f"/sys/devices/system/node/node{hbm_node}/meminfo"
    with open(meminfo) as stream:
        return next(int(line.split()[-2]) for line in stream if "MemTotal" in line)


def _apply_preferred_policy(hbm_node):
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    if hbm_node < 0 or hbm_node >= bits:
        raise RuntimeError(f"HBM node {hbm_node} does not fit in a {bits}-bit nodemask")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    nodemask = ctypes.c_ulong(1 << hbm_node)
    result = libc.syscall(
        ctypes.c_long(_syscall_number("set_mempolicy")),
        ctypes.c_int(1),
        ctypes.byref(nodemask),
        ctypes.c_ulong(hbm_node + 1),
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"set_mempolicy(MPOL_PREFERRED, node={hbm_node}) failed")


def _set_preferred(worker_id, hbm_node, required):
    try:
        total_kb = _node_capacity_kb(hbm_node)
    except (OSError, StopIteration, ValueError, IndexError) as exc:
        message = f"node={hbm_node} unavailable ({exc}); using DDR"
        if required:
            raise RuntimeError(message) from exc
        _message(worker_id, message)
        return False
    if total_kb <= 0:
        message = f"node={hbm_node} has zero capacity; using DDR"
        if required:
            raise RuntimeError(message)
        _message(worker_id, message)
        return False

    try:
        _apply_preferred_policy(hbm_node)
    except (OSError, RuntimeError) as exc:
        message = f"node={hbm_node} preference unavailable ({exc}); using DDR"
        if required:
            raise RuntimeError(message) from exc
        _message(worker_id, message)
        return False

    _message(
        worker_id,
        f"preferred node={hbm_node} capacity_mb={total_kb / 1024.0:.1f}",
    )
    return True


class HBMPlacement:
    def __init__(self, worker_id, node, base_node, required, probe=None):
        self.worker_id = worker_id
        self.node = node
        self.base_node = base_node
        self.required = required
        self._probe = probe

    def report(self):
        try:
            residency = _numa_residency_mb()
            near = residency.get(self.node, 0.0)
            hbm_total = sum(value for node, value in residency.items() if node >= self.base_node)
            ddr = sum(value for node, value in residency.items() if node < self.base_node)
            if self.worker_id == 0:
                sys.stderr.write(
                    f"HBM_SELFCHECK_MB={near:.1f} HBM_SELFCHECK_NODE={self.node} "
                    f"HBM_SELFCHECK_HBMTOTAL_MB={hbm_total:.1f} "
                    f"HBM_SELFCHECK_DDR_MB={ddr:.1f}\n"
                )
                sys.stderr.flush()
        except Exception as exc:
            if self.worker_id == 0:
                sys.stderr.write(f"HBM_SELFCHECK failed: {exc}\n")
                sys.stderr.flush()
            if self.required:
                raise
        finally:
            if self._probe is not None:
                self._probe.close()
                self._probe = None


def configure_hbm(worker_id, cpu_domain, settings: HBMSettings):
    """Configure preferred HBM and verify required mode with a controlled probe."""
    if cpu_domain < 0:
        raise ValueError("CPU domain must be non-negative")
    node = settings.base_node + cpu_domain
    if not settings.enabled:
        _message(worker_id, "disabled by WT_HBM=0; using DDR")
        return HBMPlacement(worker_id, node, settings.base_node, settings.required)

    active = _set_preferred(worker_id, node, settings.required)
    probe = (
        _create_probe(worker_id, node, settings.probe_mb) if settings.required and active else None
    )
    return HBMPlacement(worker_id, node, settings.base_node, settings.required, probe)
