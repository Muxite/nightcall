#!/usr/bin/env python3
"""
Refresh local C++ bindings and training Docker images.

Docker training images copy the repo at **build** time (``Dockerfile`` ``COPY . .``).
Edits to ``python/training/``, ``engine/``, etc. only affect containers after
``docker compose build``. Runtime bind mounts are only ``./tests``, ``./results``,
and checkpoint volumes — they do **not** sync arbitrary source edits into the image.

Typical use after changing training code or the sim:

    python scripts/refresh_training_stack.py

Skip host compile (e.g. CI machine with only Docker):

    python scripts/refresh_training_stack.py --skip-local

Rebuild images without cache:

    python scripts/refresh_training_stack.py --no-cache
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """
    Resolve repository root (directory containing ``docker-compose.yml``).

    :returns: Absolute path to the repo root.
    """
    here = Path(__file__).resolve().parent
    return here.parent


def _run(cmd: list[str], cwd: Path) -> None:
    """
    Run a command and exit the process on failure.

    :param cmd: Argument list (executable first).
    :param cwd: Working directory.
    """
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _configure_local(root: Path) -> None:
    """
    Configure CMake using the ``linux`` preset (requires ``VCPKG_ROOT``).

    :param root: Repository root.
    """
    env = os.environ.copy()
    if not env.get("VCPKG_ROOT", "").strip():
        print(
            "error: --configure-local needs VCPKG_ROOT pointing at a vcpkg clone.",
            file=sys.stderr,
        )
        sys.exit(1)
    _run(["cmake", "--preset", "linux"], cwd=root)


def _build_local(root: Path) -> None:
    """
    Build the ``nightcall_sim`` target under ``build/``.

    :param root: Repository root.
    """
    build_dir = root / "build"
    ninja = build_dir / "build.ninja"
    makefile = build_dir / "Makefile"
    if not ninja.exists() and not makefile.exists():
        print(
            "error: no configured build/ (missing build.ninja or Makefile).\n"
            "  Configure first, e.g.:\n"
            "    export VCPKG_ROOT=/path/to/vcpkg\n"
            "    cmake --preset linux\n"
            "  Or pass --configure-local with VCPKG_ROOT set.",
            file=sys.stderr,
        )
        sys.exit(1)
    nproc = os.cpu_count() or 4
    _run(["cmake", "--build", str(build_dir), "--target", "nightcall_sim", "-j", str(nproc)], cwd=root)


def _docker_compose_file(root: Path) -> Path:
    """
    Return path to ``docker-compose.yml``.

    :param root: Repository root.
    :returns: Path to the compose file.
    """
    return root / "docker-compose.yml"


def _build_docker(root: Path, which: str, no_cache: bool) -> None:
    """
    Run ``docker compose build`` for training service image(s).

    :param root: Repository root.
    :param which: ``cpu``, ``cuda``, or ``all``.
    :param no_cache: Pass ``--no-cache`` to the build.
    """
    compose = _docker_compose_file(root)
    if not compose.is_file():
        print(f"error: missing {compose}", file=sys.stderr)
        sys.exit(1)
    cmd = ["docker", "compose", "-f", str(compose), "build"]
    if no_cache:
        cmd.append("--no-cache")
    if which == "cpu":
        cmd.append("train")
    elif which == "cuda":
        cmd.append("train-cuda")
    else:
        cmd.extend(["train", "train-cuda"])
    _run(cmd, cwd=root)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rebuild nightcall_sim locally and refresh training Docker images.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/)",
    )
    p.add_argument(
        "--skip-local",
        action="store_true",
        help="Do not run cmake --build on the host",
    )
    p.add_argument(
        "--configure-local",
        action="store_true",
        help="Run cmake --preset linux before building (requires VCPKG_ROOT)",
    )
    p.add_argument(
        "--docker",
        choices=("cpu", "cuda", "all"),
        default="all",
        help="Which image(s) to rebuild (default: all)",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass --no-cache to docker compose build",
    )
    args = p.parse_args()
    root = (args.repo_root or _repo_root()).resolve()

    if args.skip_local and args.configure_local:
        print("warning: --configure-local ignored with --skip-local", flush=True)

    if not args.skip_local:
        if args.configure_local:
            _configure_local(root)
        _build_local(root)
    else:
        print("(skipped local cmake build)", flush=True)

    _build_docker(root, args.docker, args.no_cache)
    print("Done.")


if __name__ == "__main__":
    main()
