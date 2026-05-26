from __future__ import annotations

import fnmatch
import hashlib
import os
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RELEASE_NAME = "ordinal-info-processing-demo"
OUTPUT_DIR = REPO_ROOT / "release"

VERSION = "v2.7"

INCLUDE_DIRS = [
    "README.md",
    "resources",
    "model",
    "src",
]

EXCLUDE_DIRS = {
    "__pycache__",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*~",
}


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if path.name in EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_PATTERNS)


def iter_release_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_DIRS:
        path = REPO_ROOT / item
        if not path.exists():
            print(f"Warning: missing include path, skipped: {item}")
            continue
        if path.is_file():
            if not should_exclude(path):
                files.append(path)
            continue
        for root, dirnames, filenames in os.walk(path):
            root_path = Path(root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not should_exclude(root_path / dirname)
            ]
            for filename in filenames:
                file_path = root_path / filename
                if not should_exclude(file_path):
                    files.append(file_path)
    return sorted(set(files), key=lambda p: p.relative_to(REPO_ROOT).as_posix())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: list[Path], zip_name: str) -> str:
    lines = [
        f"release: {zip_name}",
        f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"file_count: {len(files)}",
        "",
        "files:",
    ]
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"- {rel}  sha256={file_sha256(path)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / f"{RELEASE_NAME}_{VERSION}.zip"
    files = iter_release_files()
    # manifest = build_manifest(files, zip_path.name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(REPO_ROOT).as_posix()
            zf.write(path, arcname)
        # zf.writestr("RELEASE_MANIFEST.txt", manifest)

    print(f"Created draft release: {zip_path}")
    print(f"Included files: {len(files)}")


if __name__ == "__main__":
    main()
