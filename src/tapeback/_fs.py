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
