"""StatusNotifierItem server — watcher registration (unit) + full lifecycle (integration)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from tests.fixtures import requires_dbus_next

pytestmark = requires_dbus_next

from tapeback._sni import StatusNotifierItem, register_with_watcher  # noqa: E402

# --- unit ---


def test_register_with_watcher_registers_service_name():
    """register_with_watcher introspects the watcher and registers our bus name."""
    iface = MagicMock()
    iface.call_register_status_notifier_item = AsyncMock()
    proxy = MagicMock()
    proxy.get_interface.return_value = iface
    bus = MagicMock()
    bus.introspect = AsyncMock(return_value="<node/>")
    bus.get_proxy_object.return_value = proxy

    asyncio.run(register_with_watcher(bus, ":1.42"))

    bus.introspect.assert_awaited_once()
    iface.call_register_status_notifier_item.assert_awaited_once_with(":1.42")


# --- integration ---


def test_sni_full_lifecycle_flow():
    """End-to-end SNI flow: idle -> recording -> attention -> idle.

    Reads every property getter, drives every public helper (which emit the
    state-change signals) and invokes every host-callable method, verifying the
    protocol contract without a live bus.
    """
    activations: list[str] = []
    sni = StatusNotifierItem(app_id="tapeback", title="tapeback", menu_path="/MenuBar")
    sni.set_callbacks(
        on_activate=lambda: activations.append("activate"),
        on_secondary_activate=lambda: activations.append("secondary"),
    )

    # Every property getter is read at least once.
    assert sni.Category == "ApplicationStatus"
    assert sni.Id == "tapeback"
    assert sni.Title == "tapeback"
    assert sni.WindowId == 0
    assert sni.Menu == "/MenuBar"
    assert sni.ItemIsMenu is False
    assert sni.IconPixmap == []
    assert sni.OverlayIconName == ""
    assert sni.OverlayIconPixmap == []
    assert sni.AttentionIconName == ""
    assert sni.AttentionIconPixmap == []
    assert sni.AttentionMovieName == ""
    assert sni.IconAccessibleDesc == ""
    assert sni.AttentionAccessibleDesc == ""
    assert sni.OverlayIconAccessibleDesc == ""
    assert sni.IconThemePath == ""

    # Idle defaults.
    assert sni.Status == "Active"
    assert sni.IconName == ""

    # Left-click activates and starts recording.
    sni.Activate(10, 20)
    assert activations == ["activate"]
    sni.set_icon("media-record-symbolic")
    sni.set_tooltip("tapeback", "Recording (standup)")
    assert sni.IconName == "media-record-symbolic"
    assert sni.ToolTip == ["", [], "tapeback", "Recording (standup)"]

    # Processing -> attention; NewStatus signal carries the new value.
    sni.set_status("NeedsAttention")
    assert sni.Status == "NeedsAttention"
    assert sni.NewStatus() == "NeedsAttention"

    # Middle-click + remaining host methods (no-ops, must not raise).
    sni.SecondaryActivate(1, 2)
    assert activations == ["activate", "secondary"]
    sni.ContextMenu(0, 0)
    sni.Scroll(1, "vertical")

    # The remaining protocol signals emit cleanly without a bus too.
    sni.NewTitle()
    sni.NewAttentionIcon()
    sni.NewOverlayIcon()

    # Back to idle.
    sni.set_icon("audio-input-microphone-symbolic")
    sni.set_status("Active")
    assert sni.IconName == "audio-input-microphone-symbolic"
    assert sni.Status == "Active"
