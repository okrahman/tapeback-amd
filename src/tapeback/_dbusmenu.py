"""com.canonical.dbusmenu D-Bus server.

Minimal flat-menu implementation. The StatusNotifierItem exposes a Menu path;
this module owns that object and serves layout / handles click events. We do
not support submenus or radio groups — tapeback's tray menu is a flat list.

Spec: https://github.com/AyatanaIndicators/libdbusmenu/blob/master/libdbusmenu-glib/dbus-menu.xml
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from dbus_next import Variant
from dbus_next.constants import PropertyAccess
from dbus_next.service import ServiceInterface, dbus_property, method, signal

DBUSMENU_INTERFACE = "com.canonical.dbusmenu"
MENU_OBJECT_PATH = "/MenuBar"

_ROOT_ID = 0  # menu root is always id 0 per spec


@dataclass
class MenuItem:
    """One flat menu entry."""

    id: int
    label: str = ""
    enabled: bool = True
    visible: bool = True
    type: str = "standard"  # "standard" or "separator"
    icon_name: str = ""
    on_clicked: Callable[[], None] | None = field(default=None, repr=False)


def _item_properties(item: MenuItem, requested: list[str]) -> dict[str, Variant]:
    """Convert a MenuItem to a dbusmenu property dict.

    If `requested` is empty, return all known props (per spec — empty filter
    means "give me everything").
    """
    full = {
        "label": Variant("s", item.label),
        "enabled": Variant("b", item.enabled),
        "visible": Variant("b", item.visible),
        "type": Variant("s", item.type),
    }
    if item.icon_name:
        full["icon-name"] = Variant("s", item.icon_name)
    if not requested:
        return full
    return {k: v for k, v in full.items() if k in requested}


def build_layout(items: list[MenuItem], property_names: list[str]) -> list:
    """Build the (id, props, children) tuple the host expects from GetLayout.

    Returned as a Python list (dbus-next marshals it to the struct signature).
    Pure function — extracted for testability without a D-Bus connection.
    """
    children = []
    for item in items:
        if not item.visible:
            continue
        children.append(
            Variant(
                "(ia{sv}av)",
                [item.id, _item_properties(item, property_names), []],
            )
        )
    root_props = {"children-display": Variant("s", "submenu")}
    return [_ROOT_ID, root_props, children]


class DBusMenu(ServiceInterface):
    """Serves the menu definition + click events for one tray item."""

    def __init__(self, items: list[MenuItem]) -> None:
        super().__init__(DBUSMENU_INTERFACE)
        self._items = items
        self._revision = 0

    # --- properties -----------------------------------------------------------

    @dbus_property(access=PropertyAccess.READ)
    def Version(self) -> "u":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return 3

    @dbus_property(access=PropertyAccess.READ)
    def TextDirection(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return "ltr"

    @dbus_property(access=PropertyAccess.READ)
    def Status(self) -> "s":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return "normal"

    @dbus_property(access=PropertyAccess.READ)
    def IconThemePath(self) -> "as":  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
        return []

    # --- methods --------------------------------------------------------------

    @method()
    def GetLayout(
        self,
        parentId: "i",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        recursionDepth: "i",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        propertyNames: "as",  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
    ) -> "u(ia{sv}av)":  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
        if parentId != _ROOT_ID:
            return [self._revision, [parentId, {}, []]]
        return [self._revision, build_layout(self._items, propertyNames)]

    @method()
    def GetGroupProperties(
        self,
        ids: "ai",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        propertyNames: "as",  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
    ) -> "a(ia{sv})":  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
        result = []
        for item_id in ids:
            item = self._find(item_id)
            if item is not None:
                result.append([item_id, _item_properties(item, propertyNames)])
        return result

    @method()
    def GetProperty(self, id: "i", name: "s") -> "v":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        item = self._find(id)
        if item is not None:
            props = _item_properties(item, [name])
            if name in props:
                return props[name]
        return Variant("s", "")

    @method()
    def Event(
        self,
        id: "i",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        eventId: "s",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        data: "v",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        timestamp: "u",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
    ) -> None:
        if eventId == "clicked":
            item = self._find(id)
            if item is not None and item.on_clicked is not None:
                item.on_clicked()

    @method()
    def EventGroup(
        self,
        events: "a(isvu)",  # type: ignore[name-defined]  # ty:ignore[invalid-type-form]
    ) -> "ai":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        missing = []
        for event in events:
            item_id, event_id, _data, _ts = event
            item = self._find(item_id)
            if item is None:
                missing.append(item_id)
                continue
            if event_id == "clicked" and item.on_clicked is not None:
                item.on_clicked()
        return missing

    @method()
    def AboutToShow(self, id: "i") -> "b":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return False  # layout always reflects current state

    @method()
    def AboutToShowGroup(
        self,
        ids: "ai",  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
    ) -> "aiai":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return [[], []]

    # --- signals --------------------------------------------------------------

    @signal()
    def LayoutUpdated(self) -> "ui":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return [self._revision, _ROOT_ID]

    @signal()
    def ItemsPropertiesUpdated(self) -> "a(ia{sv})a(ias)":  # type: ignore[name-defined]  # ty:ignore[invalid-syntax-in-forward-annotation]
        return [[], []]

    @signal()
    def ItemActivationRequested(self) -> "iu":  # type: ignore[name-defined]  # ty:ignore[unresolved-reference]
        return [0, 0]

    # --- helpers --------------------------------------------------------------

    def _find(self, item_id: int) -> MenuItem | None:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def notify_layout_changed(self) -> None:
        """Bump the revision and emit LayoutUpdated so the host re-fetches."""
        self._revision += 1
        self.LayoutUpdated()

    def replace_items(self, items: list[MenuItem]) -> None:
        """Swap the menu contents and notify the host to refetch the layout.

        Items keep stable ids — only labels / visibility / callbacks change
        across state transitions.
        """
        self._items = items
        self.notify_layout_changed()
