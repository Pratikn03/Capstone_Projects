r"""Bounded descriptor-anchored byte I/O for the later self-managed vault.

This module is deliberately not a custody broker, scientific lock, or label
authorization mechanism. It only provides a narrow primitive for synthetic
broker tests. It never follows caller-supplied descendants by pathname.

Roots use exact absolute native-string spelling; each component is opened with
no-follow relative to its verified predecessor. Names match
``[a-z0-9][a-z0-9_-]{0,95}\.json``. Renaming or replacing the root path does not
redirect the pinned descriptor. Calls must be externally serialized. These
checks do not prevent host-owner write-and-restore or mount races, establish
OS worker isolation, or prevent fork inheritance of private capabilities.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
from dataclasses import dataclass

MAX_VAULT_OBJECT_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\.json\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class VaultIOError(Exception):
    """Fixed-code error that never exposes private paths or native details."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str, cause: BaseException | None = None) -> VaultIOError:
    error = VaultIOError(code)
    if cause is not None:
        error.__cause__ = None
    return error


def _root_text(root: object) -> str:
    if type(root) is not str or not root or "\x00" in root:
        raise _fail("INVALID_ROOT")
    if not root.startswith("/") or root == "/" or "//" in root or root.endswith("/"):
        raise _fail("INVALID_ROOT")
    components = root.split("/")[1:]
    if any(part in {"", ".", ".."} for part in components):
        raise _fail("INVALID_ROOT")
    return root


def _identity(value: object) -> int:
    if type(value) is not int or value < 0:
        raise _fail("INVALID_ROOT_IDENTITY")
    return value


def _name(value: object) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise _fail("INVALID_OBJECT_NAME")
    return value


def _limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_VAULT_OBJECT_BYTES:
        raise _fail("INVALID_LIMIT")
    return value


def _digest(value: object) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise _fail("INVALID_DIGEST")
    return value


def _flags(*, write: bool = False) -> int:
    flags = (os.O_WRONLY if write else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK") and not write:
        flags |= os.O_NONBLOCK
    return flags


def _translate(exc: BaseException) -> VaultIOError:
    if isinstance(exc, VaultIOError):
        return exc
    if isinstance(exc, FileExistsError):
        return _fail("ALREADY_EXISTS")
    if isinstance(exc, PermissionError):
        return _fail("DENIED")
    if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
        return _fail("NOT_FOUND")
    if isinstance(exc, IsADirectoryError):
        return _fail("NOT_REGULAR")
    if isinstance(exc, OSError) and exc.errno == errno.ELOOP:
        return _fail("LINK_REJECTED")
    return _fail("IO_FAILURE")


def _regular_single(st: os.stat_result, *, expected_size: int | None = None) -> None:
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise _fail("NOT_SINGLE_REGULAR")
    if expected_size is not None and st.st_size != expected_size:
        raise _fail("SIZE_CHANGED")


def _snapshot(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_mode & stat.S_IFMT(st.st_mode),
        st.st_nlink,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


@dataclass
class VaultDirectory:
    _fd: int
    _device: int
    _root_inode: int
    _closed: bool = False

    @classmethod
    def open(cls, root: str, *, expected_device: int, expected_inode: int) -> VaultDirectory:
        text = _root_text(root)
        device, inode = _identity(expected_device), _identity(expected_inode)
        fd = -1
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open("/", flags)
            for component in text.split("/")[1:]:
                child = os.open(component, flags, dir_fd=fd)
                previous, fd = fd, child
                # A reported close failure has ambiguous descriptor ownership.
                # Never retry previous; the outer handler only owns child.
                try:
                    os.close(previous)
                except OSError:
                    raise _fail("CLOSE_FAILURE") from None
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode) or st.st_dev != device or st.st_ino != inode:
                raise _fail("ROOT_IDENTITY")
            os.set_inheritable(fd, False)
            return cls(fd, device, inode)
        except BaseException as exc:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    raise _fail("CLOSE_FAILURE") from None
            raise _translate(exc) from None

    def _require_open(self) -> int:
        if self._closed:
            raise _fail("CLOSED")
        return self._fd

    def close(self) -> None:
        if self._closed:
            return
        fd = self._fd
        self._closed = True
        try:
            os.close(fd)
        except OSError:
            raise _fail("CLOSE_FAILURE") from None

    def __enter__(self) -> VaultDirectory:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _close_object(self, fd: int) -> None:
        try:
            os.close(fd)
        except OSError:
            # Invalidate the capability and release its separate root descriptor.
            # The ambiguous object descriptor must never be retried.
            try:
                self.close()
            except VaultIOError:
                pass
            raise _fail("CLOSE_FAILURE") from None

    def read_bytes(self, name: str, *, expected_sha256: str, max_bytes: int) -> bytes:
        parent = self._require_open()
        leaf = _name(name)
        digest = _digest(expected_sha256)
        limit = _limit(max_bytes)
        fd = -1
        try:
            fd = os.open(leaf, _flags(), dir_fd=parent)
            before = os.fstat(fd)
            _regular_single(before)
            if before.st_size > limit:
                raise _fail("TOO_LARGE")
            chunks: list[bytes] = []
            total = 0
            while True:
                part = os.read(fd, min(READ_CHUNK_BYTES, limit + 1 - total))
                if not part:
                    break
                total += len(part)
                if total > limit:
                    raise _fail("TOO_LARGE")
                chunks.append(part)
            after = os.fstat(fd)
            _regular_single(after, expected_size=total)
            if _snapshot(before) != _snapshot(after):
                raise _fail("MUTATED")
            data = b"".join(chunks)
            if hashlib.sha256(data).hexdigest() != digest:
                raise _fail("DIGEST_MISMATCH")
            return data
        except BaseException as exc:
            raise _translate(exc) from None
        finally:
            if fd >= 0:
                self._close_object(fd)

    def create_bytes(self, name: str, data: bytes, *, max_bytes: int) -> str:
        parent = self._require_open()
        leaf = _name(name)
        limit = _limit(max_bytes)
        if type(data) is not bytes or len(data) > limit:
            raise _fail("INVALID_DATA")
        fd = -1
        try:
            fd = os.open(leaf, _flags(write=True) | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
            offset = 0
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written <= 0:
                    raise _fail("WRITE_FAILURE")
                offset += written
            final = os.fstat(fd)
            _regular_single(final, expected_size=len(data))
            if final.st_dev != self._device or stat.S_IMODE(final.st_mode) & ~0o600:
                raise _fail("WRITE_IDENTITY")
            os.fsync(fd)
            closing, fd = fd, -1
            self._close_object(closing)
            os.fsync(parent)
            return hashlib.sha256(data).hexdigest()
        except BaseException as exc:
            raise _translate(exc) from None
        finally:
            if fd >= 0:
                self._close_object(fd)
