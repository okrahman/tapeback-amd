"""Display server / desktop detection for the tray command.

Lives apart from tray.py so it can be imported (and tested) without pulling in
pystray — pystray does eager X server I/O at import time, which fails in
headless test runs and CI containers.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Desktops that route the system tray through GNOME Shell's StatusNotifier path,
# which on Wayland requires the AppIndicator Support extension. KDE/Plasma is
# excluded — Plasma's compositor speaks StatusNotifierItem natively on Wayland.
_GNOME_LIKE_DESKTOPS = frozenset({"gnome", "gnome-classic", "unity", "pantheon", "ubuntu"})

_APPINDICATOR_HINT = (
    "Tray on GNOME Wayland requires the AppIndicator Support extension.\n"
    "Install:  sudo apt install gnome-shell-extension-appindicator\n"
    'Then open the "Extensions" app, enable "Ubuntu AppIndicators" (or\n'
    '"AppIndicator and KStatusNotifierItem Support"), and re-login.\n'
    "Without it the tray icon will appear but the menu will not respond.\n"
    "Alternative: use the CLI — `tapeback start` / `tapeback stop`."
)


@dataclass(frozen=True)
class TrayEnv:
    session_type: str
    desktop: str
    needs_appindicator_hint: bool
    hint_message: str


def detect_tray_env(env: Mapping[str, str] | None = None) -> TrayEnv:
    """Inspect XDG_SESSION_TYPE / XDG_CURRENT_DESKTOP to decide if we should warn.

    Returns a hint only when the combination is known-broken without the
    AppIndicator extension (GNOME-family on Wayland). KDE Plasma Wayland is fine.
    """
    env = env if env is not None else os.environ
    session_type = env.get("XDG_SESSION_TYPE", "").lower()
    desktop = env.get("XDG_CURRENT_DESKTOP", "").lower()
    desktop_parts = {part.strip() for part in desktop.split(":") if part.strip()}
    needs_hint = session_type == "wayland" and bool(desktop_parts & _GNOME_LIKE_DESKTOPS)
    return TrayEnv(
        session_type=session_type,
        desktop=desktop,
        needs_appindicator_hint=needs_hint,
        hint_message=_APPINDICATOR_HINT if needs_hint else "",
    )
