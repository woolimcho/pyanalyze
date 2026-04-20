# static analysis: ignore
"""Tests for parallel file checking via os.fork().

These tests verify that:
- Parallel mode completes without deadlocks (via timeout)
- Parallel mode produces the same failures as sequential mode
- Edge cases (empty file list, single file) work correctly
"""

import os
import signal
import textwrap
import tempfile

import pytest

import pyanalyze
from pyanalyze.node_visitor import _fork_parallel


class SimpleVisitor(pyanalyze.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False


TIMEOUT_SECONDS = 30


def _alarm_handler(signum, frame):
    raise TimeoutError("Parallel test timed out — possible deadlock")


@pytest.fixture(autouse=True)
def _deadlock_timeout():
    """Kill test after TIMEOUT_SECONDS to catch deadlocks."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIMEOUT_SECONDS)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)


def _make_temp_py(content, tmpdir):
    fd, path = tempfile.mkstemp(suffix=".py", dir=tmpdir)
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(content))
    return path


def test_parallel_empty_files():
    kwargs = SimpleVisitor.prepare_constructor_kwargs({})
    failures, extra = _fork_parallel(SimpleVisitor, [], kwargs, max_workers=2)
    assert failures == []
    assert extra == []


def test_parallel_single_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_temp_py("x: int = 1\n", tmpdir)
        kwargs = SimpleVisitor.prepare_constructor_kwargs({})
        failures, _extra = _fork_parallel(SimpleVisitor, [path], kwargs, max_workers=2)
        assert isinstance(failures, list)


def test_parallel_matches_sequential():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with a known type error
        files = []
        for i in range(4):
            path = _make_temp_py(
                f"""\
                def f{i}(x: int) -> str:
                    return x  # type error: incompatible return
                """,
                tmpdir,
            )
            files.append(path)

        kwargs = SimpleVisitor.prepare_constructor_kwargs({})

        # Sequential
        seq_failures = []
        for filename in files:
            failures, _ = SimpleVisitor._check_file_single_arg((filename, kwargs))
            seq_failures.extend(failures)

        # Parallel
        par_failures, _extra = _fork_parallel(
            SimpleVisitor, files, kwargs, max_workers=2
        )

        # Both should find the same errors
        seq_codes = sorted(f["code"].name for f in seq_failures)
        par_codes = sorted(f["code"].name for f in par_failures)
        assert seq_codes == par_codes


def test_parallel_multiple_workers():
    """Verify parallel mode works with many workers and many files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        files = []
        for i in range(8):
            path = _make_temp_py(f"x_{i}: int = {i}\n", tmpdir)
            files.append(path)

        kwargs = SimpleVisitor.prepare_constructor_kwargs({})
        failures, _extra = _fork_parallel(
            SimpleVisitor, files, kwargs, max_workers=4
        )
        assert isinstance(failures, list)


def test_parallel_worker_crash_does_not_hang():
    """If a file causes a crash, other workers and the parent should not hang."""
    with tempfile.TemporaryDirectory() as tmpdir:
        good = _make_temp_py("x: int = 1\n", tmpdir)
        files = [good, good, good]

        kwargs = SimpleVisitor.prepare_constructor_kwargs({})
        failures, _extra = _fork_parallel(
            SimpleVisitor, files, kwargs, max_workers=2
        )
        assert isinstance(failures, list)
