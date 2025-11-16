from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import re
import tarfile
import textwrap
import zipfile

import tomllib

ROOT = Path(__file__).resolve().parent
PYPROJECT_DATA = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT_DATA["project"]
PACKAGE_MAP = PYPROJECT_DATA.get("tool", {}).get("deldel", {}).get("package-map", {})

_DISTRIBUTION = re.sub(r"[-_.]+", "_", PROJECT["name"])
_VERSION = PROJECT["version"]
_DIST_INFO = f"{_DISTRIBUTION}-{_VERSION}.dist-info"
_WHEEL_NAME = f"{_DISTRIBUTION}-{_VERSION}-py3-none-any.whl"
_SDIST_NAME = f"{PROJECT['name']}-{_VERSION}.tar.gz"


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or path.name.endswith(".pyc")


def _iter_package_files():
    for package_name, relative in sorted(PACKAGE_MAP.items()):
        src_root = ROOT / relative
        if not src_root.exists():
            continue
        for path in sorted(src_root.rglob("*")):
            if path.is_file() and not _should_skip(path):
                rel = path.relative_to(src_root)
                arcname = f"{package_name}/{rel.as_posix()}"
                yield path, arcname


def _read_long_description() -> str:
    readme_entry = PROJECT.get("readme")
    if isinstance(readme_entry, str):
        readme_path = ROOT / readme_entry
        if readme_path.exists():
            return readme_path.read_text(encoding="utf-8")
    elif isinstance(readme_entry, dict):
        path_value = readme_entry.get("file")
        if path_value:
            readme_path = ROOT / path_value
            if readme_path.exists():
                return readme_path.read_text(encoding="utf-8")
    return ""


def _metadata_text() -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {PROJECT['name']}",
        f"Version: {_VERSION}",
    ]
    if PROJECT.get("description"):
        lines.append(f"Summary: {PROJECT['description']}")
    authors = PROJECT.get("authors") or []
    if authors:
        names = ", ".join(author.get("name", "") for author in authors if author.get("name"))
        if names:
            lines.append(f"Author: {names}")
    license_info = PROJECT.get("license", {}).get("text")
    if license_info:
        lines.append(f"License: {license_info}")
    homepage = (PROJECT.get("urls") or {}).get("Homepage")
    if homepage:
        lines.append(f"Home-page: {homepage}")
    if PROJECT.get("requires-python"):
        lines.append(f"Requires-Python: {PROJECT['requires-python']}")
    readme_entry = PROJECT.get("readme")
    if readme_entry:
        content_type = "text/markdown"
        if isinstance(readme_entry, dict):
            content_type = readme_entry.get("content-type", content_type)
        lines.append(f"Description-Content-Type: {content_type}")
    for dependency in PROJECT.get("dependencies", []):
        lines.append(f"Requires-Dist: {dependency}")
    description = _read_long_description()
    if description:
        lines.append("")
        lines.append(description)
    return "\n".join(lines) + "\n"


def _wheel_metadata() -> str:
    return textwrap.dedent(
        f"""\
        Wheel-Version: 1.0
        Generator: deldel-build-backend
        Root-Is-Purelib: true
        Tag: py3-none-any
        """
    )


def _top_level_text() -> str:
    packages = sorted(PACKAGE_MAP.keys())
    return "\n".join(packages) + "\n"


def _hash_contents(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _write_dist_info() -> list[tuple[str, bytes]]:
    dist_info_entries = []
    metadata_bytes = _metadata_text().encode("utf-8")
    dist_info_entries.append((f"{_DIST_INFO}/METADATA", metadata_bytes))
    wheel_bytes = _wheel_metadata().encode("utf-8")
    dist_info_entries.append((f"{_DIST_INFO}/WHEEL", wheel_bytes))
    top_level_bytes = _top_level_text().encode("utf-8")
    dist_info_entries.append((f"{_DIST_INFO}/top_level.txt", top_level_bytes))
    return dist_info_entries


def _add_to_zip(zip_file: zipfile.ZipFile, arcname: str, data: bytes) -> tuple[str, str, int]:
    zip_file.writestr(arcname, data)
    return arcname, _hash_contents(data), len(data)


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    wheel_path = Path(wheel_directory) / _WHEEL_NAME
    records: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel_zip:
        for src, arcname in _iter_package_files():
            data = src.read_bytes()
            records.append(_add_to_zip(wheel_zip, arcname, data))
        for arcname, data in _write_dist_info():
            records.append(_add_to_zip(wheel_zip, arcname, data))
        record_path = f"{_DIST_INFO}/RECORD"
        record_lines = [f"{path},{hash_value},{size}" for path, hash_value, size in records]
        record_lines.append(f"{record_path},,")
        wheel_zip.writestr(record_path, "\n".join(record_lines) + "\n")
    return _WHEEL_NAME


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings=None) -> str:
    metadata_path = Path(metadata_directory) / _DIST_INFO
    metadata_path.mkdir(parents=True, exist_ok=True)
    (metadata_path / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (metadata_path / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8")
    (metadata_path / "top_level.txt").write_text(_top_level_text(), encoding="utf-8")
    return _DIST_INFO


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    return []


_SDIST_INCLUDE = [
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "docs",
    "src",
    "subspaces",
    "tests",
    "from_df_to_sel_time.csv",
    "experiments_outputs",
    "deldel_build_backend.py",
]


def _iter_sdist_files():
    for item in _SDIST_INCLUDE:
        path = ROOT / item
        if not path.exists():
            continue
        if path.is_file():
            yield path, Path(item)
        else:
            for sub_path in sorted(path.rglob("*")):
                if sub_path.is_file() and not _should_skip(sub_path):
                    relative = Path(item) / sub_path.relative_to(path)
                    yield sub_path, relative


def build_sdist(sdist_directory: str, config_settings=None) -> str:
    sdist_path = Path(sdist_directory) / _SDIST_NAME
    prefix = Path(f"{PROJECT['name']}-{_VERSION}")
    with tarfile.open(sdist_path, "w:gz") as tar:
        for src, relative in _iter_sdist_files():
            arcname = prefix / relative
            tar.add(src, arcname=arcname.as_posix())
    return _SDIST_NAME
