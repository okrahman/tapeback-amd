"""org.kde.StatusNotifierItem D-Bus server.

Minimal pure-Python implementation of the StatusNotifier protocol — enough
to drive a tray icon under KDE Plasma natively, and under GNOME with the
AppIndicator Support extension installed. Replaces pystray, which on Linux
defaulted to legacy XEmbed and silently failed on modern Wayland desktops.

Spec: https://www.freedesktop.org/wiki/Specifications/StatusNotifierItem/
"""

from collections.abc import Callable

from dbus_next.constants import PropertyAccess
from dbus_next.service import ServiceInterface, dbus_property, method, signal

SNI_INTERFACE = "org.kde.StatusNotifierItem"
SNI_OBJECT_PATH = "/StatusNotifierItem"
WATCHER_NAME = "org.kde.StatusNotifierWatcher"
WATCHER_PATH = "/StatusNotifierWatcher"


class StatusNotifierItem(ServiceInterface):
    """One tray icon on the session bus.

    Properties are read by the host (GNOME extension / Plasma shell) on registration
    and after Newxxx signals; methods are invoked on user input; signals are emitted
    when our state changes.
    """

    def __init__(self, *, app_id: str, title: str, menu_path: str) -> None:
        super().__init__(SNI_INTERFACE)
        self._id = app_id
        self._title = title
        self._menu_path = menu_path
        self._icon_name = ""
        self._tooltip_title = title
        self._tooltip_text = ""
        self._status = "Active"
        self._on_activate: Callable[[], None] | None = None
        self._on_secondary_activate: Callable[[], None] | None = None

    # --- properties -----------------------------------------------------------

    @dbus_property(access=PropertyAccess.READ)
    def Category(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return "ApplicationStatus"

    @dbus_property(access=PropertyAccess.READ)
    def Id(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._id

    @dbus_property(access=PropertyAccess.READ)
    def Title(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._title

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._status

    @dbus_property(access=PropertyAccess.READ)
    def WindowId(self) -> "i":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        # SNI spec says signed int. GNOME Shell rejects `u` with
        # "type u does not match expected type i" and silently hides the icon.
        return 0

    @dbus_property(access=PropertyAccess.READ)
    def IconName(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._icon_name

    @dbus_property(access=PropertyAccess.READ)
    def IconPixmap(self) -> "a(iiay)":  # type: ignore[name-defined]  # ty:ignore[invalid-type-form]
        return []

    @dbus_property(access=PropertyAccess.READ)
    def OverlayIconName(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def OverlayIconPixmap(self) -> "a(iiay)":  # type: ignore[name-defined]  # ty:ignore[invalid-type-form]
        return []

    @dbus_property(access=PropertyAccess.READ)
    def AttentionIconName(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def AttentionIconPixmap(self) -> "a(iiay)":  # type: ignore[name-defined]  # ty:ignore[invalid-type-form]
        return []

    @dbus_property(access=PropertyAccess.READ)
    def AttentionMovieName(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    # AppIndicator accessibility-description extensions to SNI. Newer GNOME
    # hosts query these and raise DBusError(UNKNOWN_PROPERTY) if missing.

    @dbus_property(access=PropertyAccess.READ)
    def IconAccessibleDesc(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def AttentionAccessibleDesc(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def OverlayIconAccessibleDesc(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def IconThemePath(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return ""

    @dbus_property(access=PropertyAccess.READ)
    def ToolTip(self) -> "(sa(iiay)ss)":  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
        # (icon_name, pixmaps, title, text)
        return ["", [], self._tooltip_title, self._tooltip_text]

    @dbus_property(access=PropertyAccess.READ)
    def ItemIsMenu(self) -> "b":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        # False → left-click triggers Activate() rather than opening the menu.
        # Hosts that ignore this still open the menu, which is fine for us.
        return False

    @dbus_property(access=PropertyAccess.READ)
    def Menu(self) -> "o":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._menu_path

    # --- methods (called by host) --------------------------------------------

    @method()
    def Activate(self, x: "i", y: "i") -> None:  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        if self._on_activate:
            self._on_activate()

    @method()
    def SecondaryActivate(self, x: "i", y: "i") -> None:  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        if self._on_secondary_activate:
            self._on_secondary_activate()

    @method()
    def ContextMenu(self, x: "i", y: "i") -> None:  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        # The host opens the DBusMenu itself; we have nothing to do here.
        pass

    @method()
    def Scroll(self, delta: "i", orientation: "s") -> None:  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        pass

    # --- signals (emitted by us) ---------------------------------------------

    @signal()
    def NewTitle(self) -> None:
        pass

    @signal()
    def NewIcon(self) -> None:
        pass

    @signal()
    def NewAttentionIcon(self) -> None:
        pass

    @signal()
    def NewOverlayIcon(self) -> None:
        pass

    @signal()
    def NewToolTip(self) -> None:
        pass

    @signal()
    def NewStatus(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return self._status

    # --- public helpers (called from app code) -------------------------------

    def set_icon(self, icon_name: str) -> None:
        self._icon_name = icon_name
        self.NewIcon()

    def set_tooltip(self, title: str, text: str = "") -> None:
        self._tooltip_title = title
        self._tooltip_text = text
        self.NewToolTip()

    def set_status(self, status: str) -> None:
        self._status = status
        self.NewStatus()

    def set_callbacks(
        self,
        on_activate: Callable[[], None] | None = None,
        on_secondary_activate: Callable[[], None] | None = None,
    ) -> None:
        self._on_activate = on_activate
        self._on_secondary_activate = on_secondary_activate


async def register_with_watcher(bus, service_name: str) -> None:
    """Tell the desktop's StatusNotifierWatcher we exist so the host (Shell)
    can lay claim to our item and start querying properties.
    """
    introspection = await bus.introspect(WATCHER_NAME, WATCHER_PATH)
    obj = bus.get_proxy_object(WATCHER_NAME, WATCHER_PATH, introspection)
    iface = obj.get_interface(WATCHER_NAME)
    await iface.call_register_status_notifier_item(service_name)
