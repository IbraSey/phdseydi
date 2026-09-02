"""Reliable execution helpers shared by the public SPIN-H experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import fcntl

from tqdm.auto import tqdm

try:
    from joblib import Parallel, delayed, parallel_config
except ImportError:  # pragma: no cover - sequential fallback remains usable
    Parallel = delayed = None

try:
    from joblib.externals.loky import get_reusable_executor
except ImportError:  # pragma: no cover - only affects explicit pool cleanup
    get_reusable_executor = None


_CALIBRATION_SEMAPHORE = None
_CALIBRATION_LOCK_PATHS = ()


def _set_calibration_semaphore(semaphore):
    global _CALIBRATION_SEMAPHORE
    _CALIBRATION_SEMAPHORE = semaphore


def _set_calibration_lock_paths(paths):
    global _CALIBRATION_LOCK_PATHS
    _CALIBRATION_LOCK_PATHS = tuple(paths)


def _acquire_calibration_file_slot(paths):
    while True:
        for path in paths:
            descriptor = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
            else:
                return descriptor
        time.sleep(0.02)


@contextmanager
def calibration_slot():
    """Limit concurrent exact GP calibrations inside a parallel campaign."""
    semaphore = _CALIBRATION_SEMAPHORE
    if semaphore is not None:
        semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()
        return
    if _CALIBRATION_LOCK_PATHS:
        descriptor = _acquire_calibration_file_slot(_CALIBRATION_LOCK_PATHS)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        return
    yield


def resolve_n_jobs(profile, n_jobs):
    """Default to one fit at a time, independently of the scientific profile."""
    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer (use -1 for all CPUs).")
    if n_jobs < 0:
        warnings.warn(
            "Negative n_jobs uses nearly/all CPUs without a memory limit. "
            "Exact GP fits can exhaust RAM; start with n_jobs=1, then test 2.",
            RuntimeWarning,
            stacklevel=2,
        )
    return int(n_jobs)


@contextmanager
def native_thread_limit():
    """Prevent nested BLAS/OpenMP/TBB parallelism, including in live kernels."""
    import openturns as ot
    from threadpoolctl import threadpool_limits

    previous = ot.TBB.GetThreadsNumber()
    ot.TBB.SetThreadsNumber(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        ot.TBB.SetThreadsNumber(previous)


def effective_worker_count(n_jobs):
    """Return the maximum number of workers represented by a joblib setting."""
    available = os.cpu_count() or 1
    if int(n_jobs) < 0:
        return max(1, available + 1 + int(n_jobs))
    return min(int(n_jobs), available)


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def checkpoint_directory(output, label, campaign, *, settings=None, source_paths=()):
    """Create a code- and configuration-specific checkpoint directory."""
    digest = hashlib.sha256()
    payload = {
        "label": str(label),
        "campaign": asdict(campaign) if is_dataclass(campaign) else campaign,
        "settings": settings or {},
    }
    digest.update(
        json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    )
    for source in sorted((Path(path) for path in source_paths), key=str):
        files = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        for path in files:
            if path.is_file():
                digest.update(str(path).encode("utf-8"))
                digest.update(path.read_bytes())
    destination = Path(output) / ".checkpoints" / str(label) / digest.hexdigest()[:16]
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _checkpoint_path(directory, task_key):
    encoded = json.dumps(task_key, sort_keys=True, default=_json_default)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return Path(directory) / f"{digest}.pkl"


def _load_checkpoint(path):
    try:
        with Path(path).open("rb") as stream:
            return True, pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False, None


def _write_checkpoint(path, result):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _execute_task(
    function, task, index, checkpoint_path, calibration_lock_paths=(),
    limit_native_threads=False,
):
    previous_paths = _CALIBRATION_LOCK_PATHS
    _set_calibration_lock_paths(calibration_lock_paths)
    try:
        if limit_native_threads:
            with native_thread_limit():
                result = function(*task)
        else:
            # OpenMP's limit is thread-local, unlike the BLAS/TBB settings
            # applied around the parent thread pool.
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1, user_api="openmp"):
                result = function(*task)
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, result)
        return index, result
    finally:
        _set_calibration_lock_paths(previous_paths)


def shutdown_process_workers():
    """Release joblib's reusable process pool between long experiment panels."""
    if get_reusable_executor is None:
        return
    try:
        executor = get_reusable_executor(reuse=True)
    except OSError:
        return
    executor.shutdown(wait=True, kill_workers=False)


def parallel_map(
    function,
    tasks,
    n_jobs,
    description,
    *,
    prefer=None,
    task_keys=None,
    checkpoint_dir=None,
    resume=True,
    max_parallel_calibrations=None,
):
    """Map tasks in stable order, optionally restoring atomic task checkpoints."""
    tasks = list(tasks)
    if task_keys is None:
        task_keys = list(range(len(tasks)))
    else:
        task_keys = list(task_keys)
    if len(task_keys) != len(tasks):
        raise ValueError("task_keys and tasks must have the same length.")
    if not isinstance(resume, bool):
        raise ValueError("resume must be boolean.")
    if max_parallel_calibrations is not None:
        if (
            isinstance(max_parallel_calibrations, bool)
            or not isinstance(max_parallel_calibrations, Integral)
            or max_parallel_calibrations < 1
        ):
            raise ValueError("max_parallel_calibrations must be a positive integer.")
        max_parallel_calibrations = int(max_parallel_calibrations)

    results = [None] * len(tasks)
    pending = []
    restored = 0
    for index, (task, task_key) in enumerate(zip(tasks, task_keys)):
        path = (
            _checkpoint_path(checkpoint_dir, task_key)
            if checkpoint_dir is not None
            else None
        )
        loaded, result = _load_checkpoint(path) if resume and path is not None else (False, None)
        if loaded:
            results[index] = result
            restored += 1
        else:
            pending.append((index, task, path))

    if restored:
        print(f"{description}: restored {restored}/{len(tasks)} task(s) from checkpoints.")
    if not pending:
        return results

    worker_count = min(effective_worker_count(n_jobs), len(pending))
    print(
        f"{description}: {worker_count if Parallel is not None else 1} worker(s), "
        "1 native thread per worker.",
        flush=True,
    )
    if worker_count == 1 or Parallel is None:
        with native_thread_limit():
            iterator = tqdm(pending, desc=description, unit="task")
            for index, task, path in iterator:
                _, results[index] = _execute_task(function, task, index, path)
        return results

    def make_jobs(lock_paths=()):
        return (
            delayed(_execute_task)(
                function, task, index, path, lock_paths,
                limit_native_threads=prefer != "threads",
            )
            for index, task, path in pending
        )

    def collect(parallel, jobs):
        completed = parallel(jobs)
        with tqdm(total=len(pending), desc=description, unit="task") as progress:
            for index, result in completed:
                results[index] = result
                progress.update()

    limit_calibrations = (
        max_parallel_calibrations is not None
        and max_parallel_calibrations < worker_count
    )
    parallel_options = {
        "n_jobs": worker_count,
        "prefer": prefer,
        "return_as": "generator_unordered",
        "pre_dispatch": worker_count,
        "batch_size": 1,
    }
    if prefer == "threads":
        previous = _CALIBRATION_SEMAPHORE
        semaphore = (
            threading.BoundedSemaphore(max_parallel_calibrations)
            if limit_calibrations
            else None
        )
        _set_calibration_semaphore(semaphore)
        try:
            # Native thread settings are process-wide, so scope them around
            # the entire thread pool rather than around individual tasks.
            with native_thread_limit():
                collect(Parallel(**parallel_options), make_jobs())
        finally:
            _set_calibration_semaphore(previous)
    else:
        try:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                if limit_calibrations:
                    with TemporaryDirectory(prefix="spinh-calibration-slots-") as directory:
                        lock_paths = tuple(
                            str(Path(directory) / f"slot-{index}.lock")
                            for index in range(max_parallel_calibrations)
                        )
                        for path in lock_paths:
                            Path(path).touch()
                        collect(Parallel(**parallel_options), make_jobs(lock_paths))
                else:
                    collect(Parallel(**parallel_options), make_jobs())
        finally:
            shutdown_process_workers()
    return results
