"""Environment loading and API key validation."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .paths import PROJECT_ROOT


PLACEHOLDER_KEY_MARKERS = ("your-", "your_", "wklej", "tutaj", "...")


def warn_if_env_file_is_too_open(env_path: Path) -> None:
    """Warn when .env has group/other permissions on POSIX systems."""
    if os.name != "posix":
        return

    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"Warning: {env_path} can be accessed by group/other users. "
            f"Secure it with: chmod 600 {env_path}",
        )


def load_env_file(env_file: Path | None = None) -> Path | None:
    """Load .env from explicit path, cwd, or project root."""
    candidates = [env_file] if env_file else [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    seen: set[Path] = set()

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if not candidate.exists():
            if env_file:
                raise FileNotFoundError(f"Env file does not exist: {candidate}")
            continue

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("Missing package python-dotenv. Install dependencies with: pip install -r requirements.txt") from exc

        load_dotenv(candidate, override=False)
        warn_if_env_file_is_too_open(candidate)
        return candidate

    return None


def looks_like_placeholder(value: str | None) -> bool:
    """Return true when an API key is missing or appears to be a placeholder."""
    if not value:
        return True
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_KEY_MARKERS)


def get_required_api_key(name: str, loaded_env: Path | None = None) -> str:
    """Read a required API key from the environment."""
    value = os.getenv(name)
    if not looks_like_placeholder(value):
        return str(value)

    location_hint = loaded_env or (PROJECT_ROOT / ".env")
    raise RuntimeError(f"{name} is missing or still looks like a placeholder. Add it to {location_hint}.")
