import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import IO, Any

try:
    import psutil
except ImportError:  # pragma: no cover - dependency is included in release builds
    psutil = None  # type: ignore[assignment]


PROTOCOL_PREFIX = "__UFR_SCAN_WORKER_V1__"
DEFAULT_STALL_SECONDS = 300.0
DEFAULT_MAX_JOBS_PER_WORKER = 512


def encode_protocol_message(payload: dict[str, Any]) -> str:
    """Encode one worker-protocol frame."""
    return PROTOCOL_PREFIX + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_protocol_message(line: str) -> dict[str, Any] | None:
    """Decode a worker frame, ignoring ordinary log lines."""
    stripped = line.rstrip("\r\n")
    if not stripped.startswith(PROTOCOL_PREFIX):
        return None
    try:
        payload = json.loads(stripped[len(PROTOCOL_PREFIX) :])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_protocol_message(stream: IO[str], payload: dict[str, Any]) -> None:
    """Write and flush one worker-protocol frame."""
    stream.write(encode_protocol_message(payload) + "\n")
    stream.flush()


@dataclass(frozen=True)
class ActivitySnapshot:
    """Monotonic process-tree activity counters."""

    cpu_seconds: float
    io_bytes: int


class WorkerActivityTracker:
    """Track inactivity without imposing a total wall-clock deadline."""

    def __init__(
        self,
        *,
        stall_seconds: float,
        now: float,
        initial_snapshot: ActivitySnapshot | None,
    ) -> None:
        self.stall_seconds = max(0.0, float(stall_seconds))
        self._last_snapshot = initial_snapshot
        self._last_activity_at = float(now)
        self._sample_available = initial_snapshot is not None

    def observe(
        self,
        snapshot: ActivitySnapshot | None,
        *,
        now: float,
    ) -> bool:
        """Record sampled CPU/I/O counters and return whether they changed."""
        if snapshot is None:
            self._sample_available = False
            return False

        changed = self._last_snapshot is None or snapshot != self._last_snapshot
        self._last_snapshot = snapshot
        self._sample_available = True
        if changed:
            self._last_activity_at = float(now)
        return changed

    def record_protocol_activity(self, *, now: float) -> None:
        """Record a progress frame emitted by the worker."""
        self._last_activity_at = float(now)

    def idle_seconds(self, *, now: float) -> float:
        return max(0.0, float(now) - self._last_activity_at)

    def is_stalled(self, *, now: float) -> bool:
        if self.stall_seconds <= 0 or not self._sample_available:
            return False
        return self.idle_seconds(now=now) >= self.stall_seconds


def sample_process_tree_activity(pid: int) -> ActivitySnapshot | None:
    """Return aggregate CPU and I/O counters for a worker process tree."""
    if psutil is None:
        return None
    try:
        root = psutil.Process(int(pid))
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError, ValueError):
        return None

    cpu_seconds = 0.0
    io_bytes = 0
    sampled = False
    for process in processes:
        with suppress(psutil.Error, OSError, ValueError):
            cpu_times = process.cpu_times()
            cpu_seconds += float(cpu_times.user) + float(cpu_times.system)
            sampled = True
        with suppress(psutil.Error, OSError, ValueError):
            io_counters = process.io_counters()
            io_bytes += int(io_counters.read_bytes) + int(io_counters.write_bytes)
            sampled = True
    if not sampled:
        return None
    return ActivitySnapshot(cpu_seconds=cpu_seconds, io_bytes=io_bytes)


def _exit_code_hint(exit_code: int | None) -> str | None:
    if exit_code in {-1073741819, 3221225477}:
        return "ACCESS_VIOLATION(0xC0000005)"
    return None


class ScanWorkerFailure(RuntimeError):
    """Base class for failures that invalidate a persistent worker."""

    kind = "worker_failure"

    def __init__(
        self,
        message: str,
        *,
        asset_path: str,
        worker_id: int,
        pid: int | None,
    ) -> None:
        super().__init__(message)
        self.asset_path = asset_path
        self.worker_id = int(worker_id)
        self.pid = pid


class ScanWorkerCrashed(ScanWorkerFailure):
    """A worker exited or closed its response pipe during an assigned job."""

    kind = "crashed"

    def __init__(
        self,
        *,
        asset_path: str,
        worker_id: int,
        pid: int | None,
        exit_code: int | None,
        output_tail: str = "",
    ) -> None:
        self.exit_code = exit_code
        hint = _exit_code_hint(exit_code)
        exit_text = "unknown" if exit_code is None else str(exit_code)
        if hint:
            exit_text += f" [{hint}]"
        detail = f"; output={output_tail}" if output_tail else ""
        super().__init__(
            f"scan worker hard crash: file={asset_path}; exit={exit_text}{detail}",
            asset_path=asset_path,
            worker_id=worker_id,
            pid=pid,
        )


class ScanWorkerStalled(ScanWorkerFailure):
    """A live worker showed no CPU, I/O, or protocol activity."""

    kind = "stalled"

    def __init__(
        self,
        *,
        asset_path: str,
        worker_id: int,
        pid: int | None,
        idle_seconds: float,
    ) -> None:
        self.idle_seconds = float(idle_seconds)
        super().__init__(
            f"scan worker stalled: file={asset_path}; "
            f"inactive={self.idle_seconds:.1f}s",
            asset_path=asset_path,
            worker_id=worker_id,
            pid=pid,
        )


class ScanWorkerProtocolError(ScanWorkerFailure):
    """The worker stayed alive but returned an invalid protocol response."""

    kind = "protocol_error"


@dataclass
class ScanPoolResult:
    index: int
    asset_path: str
    payload: dict[str, Any]
    error: str | None
    warning: str | None
    attempts: int
    recovered_failure_kind: str | None = None


class PersistentScanWorkerSession:
    """One long-lived scan subprocess that handles jobs sequentially."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        worker_id: int,
        startup_timeout: float = 0.0,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        poll_interval: float = 1.0,
        max_jobs: int = DEFAULT_MAX_JOBS_PER_WORKER,
        activity_sampler: Callable[[int], ActivitySnapshot | None] = (
            sample_process_tree_activity
        ),
        clock: Callable[[], float] = time.monotonic,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.command = [str(part) for part in command]
        self.worker_id = int(worker_id)
        # Optional hard startup deadline. Production leaves this disabled and
        # uses the same CPU/I/O inactivity rule as file scans.
        self.startup_timeout = max(0.0, float(startup_timeout))
        self.stall_seconds = max(0.0, float(stall_seconds))
        self.poll_interval = max(0.01, float(poll_interval))
        self.max_jobs = max(1, int(max_jobs))
        self._activity_sampler = activity_sampler
        self._clock = clock
        self._popen_factory = popen_factory

        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._output_tail: deque[str] = deque(maxlen=40)
        self._output_lock = threading.Lock()
        self._jobs_completed = 0
        self._current_asset_path: str | None = None

    @property
    def pid(self) -> int | None:
        process = self._process
        return int(process.pid) if process is not None else None

    def _tail_text(self) -> str:
        with self._output_lock:
            return " | ".join(self._output_tail)

    def _read_output(
        self,
        process: subprocess.Popen[str],
        messages: queue.Queue[dict[str, Any]],
        output_tail: deque[str],
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            messages.put({"type": "_eof"})
            return
        try:
            for line in stdout:
                payload = decode_protocol_message(line)
                if payload is not None:
                    messages.put(payload)
                    continue
                text = line.strip()
                if text:
                    with self._output_lock:
                        output_tail.append(text)
        finally:
            # Keep each reader bound to the queue created for its own process.
            # A late EOF from a recycled worker must never poison its successor.
            messages.put({"type": "_eof"})

    def _spawn(self, *, asset_path: str) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return

        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        messages: queue.Queue[dict[str, Any]] = queue.Queue()
        output_tail: deque[str] = deque(maxlen=40)
        self._messages = messages
        self._output_tail = output_tail
        try:
            process = self._popen_factory(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
            )
        except Exception as error:
            raise ScanWorkerProtocolError(
                f"failed to start scan worker: file={asset_path}; {error!r}",
                asset_path=asset_path,
                worker_id=self.worker_id,
                pid=None,
            ) from error

        self._process = process
        self._jobs_completed = 0
        self._reader_thread = threading.Thread(
            target=self._read_output,
            args=(process, messages, output_tail),
            name=f"scan-worker-output-{self.worker_id}",
            daemon=True,
        )
        self._reader_thread.start()

        started_at = self._clock()
        tracker = WorkerActivityTracker(
            stall_seconds=self.stall_seconds,
            now=started_at,
            initial_snapshot=self._activity_sampler(process.pid),
        )
        while True:
            now = self._clock()
            elapsed = now - started_at
            if self.startup_timeout > 0 and elapsed >= self.startup_timeout:
                pid = self.pid
                self.close(force=True)
                raise ScanWorkerProtocolError(
                    f"scan worker startup timed out: file={asset_path}; "
                    f"limit={self.startup_timeout:.1f}s",
                    asset_path=asset_path,
                    worker_id=self.worker_id,
                    pid=pid,
                )
            try:
                wait_seconds = self.poll_interval
                if self.startup_timeout > 0:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.01, self.startup_timeout - elapsed),
                    )
                message = self._messages.get(timeout=wait_seconds)
            except queue.Empty:
                if process.poll() is not None:
                    raise ScanWorkerCrashed(
                        asset_path=asset_path,
                        worker_id=self.worker_id,
                        pid=self.pid,
                        exit_code=process.returncode,
                        output_tail=self._tail_text(),
                    )
                now = self._clock()
                tracker.observe(
                    self._activity_sampler(process.pid),
                    now=now,
                )
                if tracker.is_stalled(now=now):
                    raise ScanWorkerStalled(
                        asset_path=asset_path,
                        worker_id=self.worker_id,
                        pid=process.pid,
                        idle_seconds=tracker.idle_seconds(now=now),
                    )
                continue

            message_type = message.get("type")
            if message_type == "ready":
                return
            if message_type == "activity":
                tracker.record_protocol_activity(now=self._clock())
                continue
            if message_type == "fatal":
                detail = str(message.get("error", "worker initialization failed"))
                raise ScanWorkerProtocolError(
                    f"scan worker initialization failed: file={asset_path}; {detail}",
                    asset_path=asset_path,
                    worker_id=self.worker_id,
                    pid=self.pid,
                )
            if message_type == "_eof":
                with suppress(subprocess.TimeoutExpired, OSError):
                    process.wait(timeout=0.5)
                raise ScanWorkerCrashed(
                    asset_path=asset_path,
                    worker_id=self.worker_id,
                    pid=self.pid,
                    exit_code=process.poll(),
                    output_tail=self._tail_text(),
                )

    def _send(self, payload: dict[str, Any], *, asset_path: str) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ScanWorkerCrashed(
                asset_path=asset_path,
                worker_id=self.worker_id,
                pid=self.pid,
                exit_code=process.poll() if process is not None else None,
                output_tail=self._tail_text(),
            )
        try:
            write_protocol_message(process.stdin, payload)
        except (BrokenPipeError, OSError, ValueError) as error:
            raise ScanWorkerCrashed(
                asset_path=asset_path,
                worker_id=self.worker_id,
                pid=self.pid,
                exit_code=process.poll(),
                output_tail=self._tail_text() or repr(error),
            ) from error

    def scan(self, job_id: int, asset_path: str) -> dict[str, Any]:
        """Scan one file and return its result payload."""
        normalized_path = str(asset_path)
        self._current_asset_path = normalized_path
        self._spawn(asset_path=normalized_path)
        process = self._process
        assert process is not None

        self._send(
            {
                "type": "scan",
                "job_id": int(job_id),
                "path": normalized_path,
            },
            asset_path=normalized_path,
        )
        now = self._clock()
        tracker = WorkerActivityTracker(
            stall_seconds=self.stall_seconds,
            now=now,
            initial_snapshot=self._activity_sampler(process.pid),
        )

        try:
            while True:
                try:
                    message = self._messages.get(timeout=self.poll_interval)
                except queue.Empty:
                    if process.poll() is not None:
                        raise ScanWorkerCrashed(
                            asset_path=normalized_path,
                            worker_id=self.worker_id,
                            pid=process.pid,
                            exit_code=process.returncode,
                            output_tail=self._tail_text(),
                        )
                    now = self._clock()
                    tracker.observe(
                        self._activity_sampler(process.pid),
                        now=now,
                    )
                    if tracker.is_stalled(now=now):
                        raise ScanWorkerStalled(
                            asset_path=normalized_path,
                            worker_id=self.worker_id,
                            pid=process.pid,
                            idle_seconds=tracker.idle_seconds(now=now),
                        )
                    continue

                message_type = message.get("type")
                if message_type == "_eof":
                    with suppress(subprocess.TimeoutExpired, OSError):
                        process.wait(timeout=0.5)
                    raise ScanWorkerCrashed(
                        asset_path=normalized_path,
                        worker_id=self.worker_id,
                        pid=process.pid,
                        exit_code=process.poll(),
                        output_tail=self._tail_text(),
                    )

                if message.get("job_id") != int(job_id):
                    continue
                if message_type == "activity":
                    tracker.record_protocol_activity(now=self._clock())
                    continue
                if message_type != "result":
                    continue

                payload = message.get("payload")
                if not isinstance(payload, dict):
                    raise ScanWorkerProtocolError(
                        f"invalid scan worker result: file={normalized_path}",
                        asset_path=normalized_path,
                        worker_id=self.worker_id,
                        pid=process.pid,
                    )
                self._jobs_completed += 1
                if self._jobs_completed >= self.max_jobs:
                    self.close()
                return payload
        finally:
            self._current_asset_path = None

    def restart(self) -> None:
        """Discard the current process; the next scan starts a clean worker."""
        self.close(force=True)

    def _kill_process_tree(self, process: subprocess.Popen[str]) -> None:
        if psutil is not None:
            with suppress(psutil.Error, OSError, ValueError):
                root = psutil.Process(process.pid)
                children = root.children(recursive=True)
                for child in reversed(children):
                    with suppress(psutil.Error, OSError):
                        child.kill()
        with suppress(OSError):
            process.kill()

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and not force:
            try:
                self._send({"type": "shutdown"}, asset_path="<shutdown>")
                process.wait(timeout=3.0)
            except (ScanWorkerFailure, subprocess.TimeoutExpired, OSError):
                force = True
        if process.poll() is None and force:
            self._kill_process_tree(process)
            with suppress(subprocess.TimeoutExpired, OSError):
                process.wait(timeout=3.0)
        for stream in (process.stdin, process.stdout):
            if stream is None:
                continue
            with suppress(OSError, ValueError):
                stream.close()
        reader = self._reader_thread
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.5)
        self._process = None
        self._reader_thread = None
        self._jobs_completed = 0


class PersistentScanWorkerPool:
    """Bounded pool of long-lived worker sessions."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        worker_count: int,
        max_retries: int = 1,
        startup_timeout: float = 0.0,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        poll_interval: float = 1.0,
        max_jobs_per_worker: int = DEFAULT_MAX_JOBS_PER_WORKER,
        session_factory: Callable[[int], Any] | None = None,
        progress_callback: Callable[[int, int, ScanPoolResult], None] | None = None,
    ) -> None:
        self.command = [str(part) for part in command]
        self.worker_count = max(1, int(worker_count))
        self.max_retries = max(0, int(max_retries))
        self.startup_timeout = float(startup_timeout)
        self.stall_seconds = float(stall_seconds)
        self.poll_interval = float(poll_interval)
        self.max_jobs_per_worker = int(max_jobs_per_worker)
        self._session_factory = session_factory
        self._progress_callback = progress_callback

    def _make_session(self, worker_id: int) -> Any:
        if self._session_factory is not None:
            return self._session_factory(worker_id)
        return PersistentScanWorkerSession(
            self.command,
            worker_id=worker_id,
            startup_timeout=self.startup_timeout,
            stall_seconds=self.stall_seconds,
            poll_interval=self.poll_interval,
            max_jobs=self.max_jobs_per_worker,
        )

    def scan(self, asset_paths: Sequence[str]) -> list[ScanPoolResult]:
        paths = [str(path) for path in asset_paths]
        if not paths:
            return []

        work: queue.Queue[tuple[int, str] | None] = queue.Queue()
        for index, path in enumerate(paths):
            work.put((index, path))
        worker_total = min(self.worker_count, len(paths))
        for _ in range(worker_total):
            work.put(None)

        results: list[ScanPoolResult | None] = [None] * len(paths)
        result_lock = threading.Lock()
        session_lock = threading.Lock()
        active_sessions: list[Any] = []
        completed = 0

        def worker_loop(worker_id: int) -> None:
            nonlocal completed
            session = self._make_session(worker_id)
            with session_lock:
                active_sessions.append(session)
            try:
                while True:
                    item = work.get()
                    if item is None:
                        work.task_done()
                        break
                    index, asset_path = item
                    attempts = 0
                    recovered_kind: str | None = None
                    recovery_warning: str | None = None
                    payload: dict[str, Any] = {"ttf": [], "sdf": [], "error": None}
                    final_error: str | None = None

                    while True:
                        attempts += 1
                        try:
                            payload = session.scan(index, asset_path)
                            break
                        except ScanWorkerFailure as failure:
                            if attempts <= self.max_retries:
                                recovered_kind = failure.kind
                                recovery_warning = (
                                    f"{failure}; retrying on a clean worker"
                                )
                                with suppress(Exception):
                                    session.restart()
                                continue
                            final_error = str(failure)
                            with suppress(Exception):
                                session.restart()
                            break
                        except Exception as error:  # noqa: BLE001
                            final_error = (
                                f"scan worker internal failure: file={asset_path}; "
                                f"{error!r}"
                            )
                            with suppress(Exception):
                                session.restart()
                            break

                    result = ScanPoolResult(
                        index=index,
                        asset_path=asset_path,
                        payload=payload,
                        error=final_error,
                        warning=recovery_warning if final_error is None else None,
                        attempts=attempts,
                        recovered_failure_kind=(
                            recovered_kind if final_error is None else None
                        ),
                    )
                    with result_lock:
                        results[index] = result
                        completed += 1
                        completed_now = completed
                    if self._progress_callback is not None:
                        self._progress_callback(completed_now, len(paths), result)
                    work.task_done()
            finally:
                with suppress(Exception):
                    session.close()
                with session_lock:
                    if session in active_sessions:
                        active_sessions.remove(session)

        threads = [
            threading.Thread(
                target=worker_loop,
                args=(worker_id,),
                name=f"scan-worker-slot-{worker_id}",
                daemon=True,
            )
            for worker_id in range(worker_total)
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except BaseException:
            # A manual Ctrl+C must not leave frozen worker executables running
            # after the parent exits.
            with session_lock:
                sessions_to_close = list(active_sessions)
            for session in sessions_to_close:
                try:
                    session.close(force=True)
                except TypeError:
                    with suppress(Exception):
                        session.close()
                except Exception:  # noqa: BLE001, S112
                    continue
            for thread in threads:
                thread.join(timeout=5.0)
            raise

        normalized_results: list[ScanPoolResult] = []
        for index, result in enumerate(results):
            if result is None:
                result = ScanPoolResult(
                    index=index,
                    asset_path=paths[index],
                    payload={"ttf": [], "sdf": [], "error": None},
                    error=(
                        f"scan worker internal failure: file={paths[index]}; "
                        "worker thread ended without a result"
                    ),
                    warning=None,
                    attempts=0,
                )
            normalized_results.append(result)
        return normalized_results
