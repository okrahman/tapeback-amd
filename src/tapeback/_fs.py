"""Filesystem safety helpers for staging and recording directories.

Staging and recording files live under predictable paths (/tmp/tapeback/...),
which is required for stable resume-cache identities but means an unrelated
local process can pre-create those paths. mkdir(mode=0o700, exist_ok=True)
does NOT defend against that: an existing directory is accepted with whatever
mode and owner it already has, and fixed filenames inside it are followed
through planted symlinks. These helpers verify instead of assuming.
"""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

_PRIVATE_DIR_MODE = 0o700


def ensure_private_dir(path: Path) -> None:
    """Create `path` as a private 0700 directory, or verify an existing one.

    - missing: created with mode 0700 (subject to no umask surprises we care
      about, since the verification below repairs the mode anyway);
    - existing real directory owned by the current user: repaired to 0700,
      because the mode is the property we actually rely on;
    - a symlink or a directory owned by someone else: refused — either means
      the path was placed there by something other than tapeback.
    """
    if path.is_symlink() or path.exists():
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise RuntimeError(
                f"Refusing unsafe directory {path}: it exists but is not a real directory"
            )
        if st.st_uid != os.getuid():
            raise RuntimeError(
                f"Refusing unsafe directory {path}: it is not owned by the current user"
            )
        if stat.S_IMODE(st.st_mode) != _PRIVATE_DIR_MODE:
            path.chmod(_PRIVATE_DIR_MODE)
    else:
        path.mkdir(mode=_PRIVATE_DIR_MODE, parents=True)


def refuse_symlink_target(path: Path, purpose: str) -> None:
    """Raise unless `path` is safe to write at its fixed location.

    A symlink at a predictable path would make a writer clobber or expose a
    file the tapeback process can otherwise reach. Only a planted symlink is
    refused; stale regular files are the caller's to overwrite or remove.
    """
    if path.is_symlink():
        raise RuntimeError(f"Refusing to {purpose} through symlink: {path}")


def require_fresh_regular_target(path: Path, purpose: str) -> None:
    """Ensure `path` is a fresh, private regular file this process creates.

    Callers write recording and staging output to predictable paths inside a
    directory that `ensure_private_dir` has just secured. Refusing only
    symlinks was not enough: the directory may have been permissive until a
    moment ago, and an attacker who planted a FIFO — or who holds an open
    descriptor on a planted regular file — from before the chmod keeps a live
    handle that reads everything the writer produces afterwards. So:

    - a symlink is refused outright (as before);
    - a directory at the path is refused (it cannot be unlinked away);
    - any other pre-existing inode (FIFO, socket, device, regular file) is
      unlinked, redirecting every subsequent write to a fresh inode that the
      attacker's stale descriptor cannot follow;
    - the file is then pre-created with O_CREAT|O_EXCL and mode 0600, closing
      the race between the check and the writer's open: the writer opens and
      truncates a file only this user could have created inside a 0700
      directory.
    """
    if path.is_symlink():
        raise RuntimeError(f"Refusing to {purpose} through symlink: {path}")
    try:
        st = path.lstat()
    except FileNotFoundError:
        st = None
    if st is not None:
        if stat.S_ISDIR(st.st_mode):
            raise RuntimeError(f"Refusing to {purpose}: a directory exists at {path}")
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(fd)


def write_private_text(path: Path, text: str) -> None:
    """Atomically write `text` to a 0600 file inside a verified 0700 directory.

    Transcript-adjacent files (resume-cache entries, run records, session
    state) hold the same sensitive content the staging helpers protect, so
    they get the same treatment — plus two properties those helpers do not
    need:

    - private by default: created O_CREAT|O_EXCL mode 0600, never subject to
      a permissive umask;
    - atomic: the payload is written to a temporary file in the same
      directory and `os.replace`d into place, so a crash mid-write can never
      leave a truncated file at the real path (a reader sees either the old
      entry or the new one, never a half-written one).
    """
    ensure_private_dir(path.parent)
    # A uuid4 fragment on top of the pid: a previous crash can leave
    # .<name>.tmp.<pid> behind, and pid reuse would then hit the O_EXCL open
    # failure on a pid-only name — a failure callers (resume cache, run log)
    # swallow, silently skipping the write. A unique name cannot collide.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
