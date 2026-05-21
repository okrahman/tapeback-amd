"""Regression tests for issue #3 follow-up — SNI protocol compliance.

The original tray rewrite (v0.9.4) registered with the watcher but the icon
never appeared on GNOME Wayland because:
  1. WindowId was exposed as `u` (unsigned) — GNOME Shell rejects with
     "Received property WindowId with type u does not match expected type i".
  2. IconAccessibleDesc / AttentionAccessibleDesc / OverlayIconAccessibleDesc
     properties were missing — GNOME queries them and dbus-next answers
     `DBusError: interface ... does not have property "IconAccessibleDesc"`.
  3. The hint message was printed AND logged, producing a duplicate.
  4. Hint message was Ubuntu-only.

These tests lock in the fixes so the same regressions can't return.
"""

import logging

from dbus_next.service import ServiceInterface

from tests.fixtures import requires_dbus_next

pytestmark = requires_dbus_next

from tapeback._sni import StatusNotifierItem  # noqa: E402
from tapeback._tray_env import detect_tray_env  # noqa: E402
from tapeback.tray import _warn_if_tray_host_missing  # noqa: E402


def _properties_by_name(iface: ServiceInterface) -> dict:
    return {p.name: p for p in ServiceInterface._get_properties(iface)}


def test_window_id_is_signed_int():
    """SNI spec defines WindowId as signed int (i). GNOME rejects unsigned."""
    sni = StatusNotifierItem(app_id="x", title="x", menu_path="/MenuBar")
    props = _properties_by_name(sni)
    assert props["WindowId"].signature == "i"


def test_icon_accessible_desc_property_present():
    """GNOME's AppIndicator extension queries IconAccessibleDesc."""
    sni = StatusNotifierItem(app_id="x", title="x", menu_path="/MenuBar")
    props = _properties_by_name(sni)
    assert "IconAccessibleDesc" in props
    assert props["IconAccessibleDesc"].signature == "s"


def test_attention_accessible_desc_property_present():
    sni = StatusNotifierItem(app_id="x", title="x", menu_path="/MenuBar")
    props = _properties_by_name(sni)
    assert "AttentionAccessibleDesc" in props
    assert props["AttentionAccessibleDesc"].signature == "s"


def test_overlay_icon_accessible_desc_property_present():
    sni = StatusNotifierItem(app_id="x", title="x", menu_path="/MenuBar")
    props = _properties_by_name(sni)
    assert "OverlayIconAccessibleDesc" in props
    assert props["OverlayIconAccessibleDesc"].signature == "s"


def test_warn_emits_single_logger_record_no_stderr_print(caplog, capsys):
    """The tray-host warning must be logged once and NOT also printed.

    v0.9.4 did both, doubling the message in user logs.
    """
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"})
    with caplog.at_level(logging.WARNING, logger="tapeback.tray"):
        _warn_if_tray_host_missing(env)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_skipped_when_hint_not_needed(caplog):
    env = detect_tray_env({"XDG_SESSION_TYPE": "x11", "XDG_CURRENT_DESKTOP": "GNOME"})
    with caplog.at_level(logging.WARNING, logger="tapeback.tray"):
        _warn_if_tray_host_missing(env)
    assert not caplog.records


def test_hint_message_is_distro_neutral():
    """Runtime warning lists install paths for more than just Ubuntu."""
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"})
    assert "apt" in env.hint_message
    assert "dnf" in env.hint_message
