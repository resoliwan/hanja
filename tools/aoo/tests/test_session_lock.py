"""Session locks survive abrupt exits but release after orderly cleanup."""

import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sessions import _lock


class SessionLockTest(unittest.TestCase):
    def test_normal_cleanup_releases_lock_inherited_by_device_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supervisor.lock"
            helper = None
            try:
                with _lock(path) as handle:
                    helper = subprocess.Popen(
                        [sys.executable, "-c", "import sys; sys.stdin.read()"],
                        stdin=subprocess.PIPE, pass_fds=(handle.fileno(),),
                    )
                    with path.open("a") as contender:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertIsNone(helper.poll())
                with _lock(path, blocking=False):
                    pass
            finally:
                if helper is not None:
                    helper.communicate(timeout=5)
