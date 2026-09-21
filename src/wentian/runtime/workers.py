"""Multiprocess worker lifecycle with bounded barrier waits."""

import threading
import time
from contextlib import suppress


class WorkerGroup:
    """Own worker startup, barrier waits, and deterministic shutdown."""

    def __init__(self, processes, control, worker_barrier, io_barrier, timeout_seconds):
        self.processes = processes
        self.control = control
        self.worker_barrier = worker_barrier
        self.io_barrier = io_barrier
        self.timeout_seconds = timeout_seconds
        self.closed = False
        self._started = []

    def __enter__(self):
        try:
            for process in self.processes:
                process.start()
                self._started.append(process)
        except BaseException:
            self.closed = True
            self._abort()
            self._join_or_terminate()
            raise
        return self

    def wait(self):
        try:
            self.io_barrier.wait()
        except threading.BrokenBarrierError as exc:
            self._abort()
            raise RuntimeError(self._failure_message("worker barrier failed")) from exc

    def _abort(self):
        with suppress(Exception):
            self.worker_barrier.abort()
        with suppress(Exception):
            self.io_barrier.abort()

    def _close(self):
        if self.closed:
            return
        self.closed = True
        self.control[0] = 0
        barrier_error = None
        try:
            self.wait()
        except RuntimeError as exc:
            barrier_error = exc
        self._join_or_terminate()
        failures = [process for process in self._started if process.exitcode != 0]
        if failures:
            raise RuntimeError(self._failure_message("worker shutdown failed"))
        if barrier_error is not None:
            raise barrier_error

    def _join_or_terminate(self):
        deadline = time.monotonic() + self.timeout_seconds
        for process in self._started:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [process for process in self._started if process.is_alive()]
        for process in alive:
            process.terminate()
        deadline = time.monotonic() + 10.0
        for process in alive:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        stubborn = [process for process in alive if process.is_alive()]
        for process in stubborn:
            process.kill()
        for process in stubborn:
            process.join(timeout=10.0)

    def _failure_message(self, prefix):
        states = ", ".join(
            f"pid={process.pid} exitcode={process.exitcode} alive={process.is_alive()}"
            for process in self._started
        )
        return f"{prefix}: {states}"

    def __exit__(self, exc_type, exc_value, traceback_value):
        if exc_type is not None:
            self.closed = True
            self._abort()
            self._join_or_terminate()
            return False
        self._close()
        return False
