"""DBusMenu server + layout/property helpers — exercised without a live D-Bus connection."""

from tests.fixtures import requires_dbus_next

pytestmark = requires_dbus_next

from dbus_next import Variant  # noqa: E402

from tapeback._dbusmenu import DBusMenu, MenuItem, _item_properties, build_layout  # noqa: E402


def test_item_properties_full_set_when_no_filter():
    item = MenuItem(id=1, label="Hello", icon_name="media-record-symbolic")
    props = _item_properties(item, [])
    assert set(props) == {"label", "enabled", "visible", "type", "icon-name"}
    assert props["label"].value == "Hello"


def test_item_properties_filters_to_requested():
    item = MenuItem(id=1, label="Hello", enabled=False)
    props = _item_properties(item, ["label", "enabled"])
    assert set(props) == {"label", "enabled"}
    assert props["enabled"].value is False


def test_item_properties_omits_icon_when_empty():
    item = MenuItem(id=1, label="Hello")
    props = _item_properties(item, [])
    assert "icon-name" not in props


def test_build_layout_skips_invisible_items():
    items = [
        MenuItem(id=1, label="Shown", visible=True),
        MenuItem(id=2, label="Hidden", visible=False),
        MenuItem(id=3, label="AlsoShown", visible=True),
    ]
    root_id, _root_props, children = build_layout(items, [])
    assert root_id == 0
    assert len(children) == 2
    assert children[0].value[0] == 1
    assert children[1].value[0] == 3


def test_build_layout_root_has_submenu_children_display():
    items = [MenuItem(id=1, label="X")]
    _root_id, root_props, _children = build_layout(items, [])
    assert root_props["children-display"].value == "submenu"


def test_build_layout_propagates_property_filter():
    items = [MenuItem(id=1, label="Hello", icon_name="media-record-symbolic")]
    _root_id, _root_props, children = build_layout(items, ["label"])
    child_id, child_props, child_children = children[0].value
    assert child_id == 1
    assert set(child_props) == {"label"}
    assert child_children == []


def _call(menu, name, *args):
    """Invoke a @method via its underlying function.

    dbus-next's @method wrapper runs the body but swallows the return value on
    direct Python calls, so reach __wrapped__ to assert the protocol contract.
    """
    return getattr(menu, name).__wrapped__(menu, *args)


def test_dbusmenu_full_lifecycle_flow():
    """End-to-end DBusMenu flow: serve properties/layout, dispatch clicks, swap
    items on a state change — exercising the server without a live bus.
    """
    clicks: list[int] = []
    items = [
        MenuItem(id=1, label="Start Recording", on_clicked=lambda: clicks.append(1)),
        MenuItem(id=2, label="Quit", on_clicked=lambda: clicks.append(2)),
        MenuItem(id=3, type="separator"),
    ]
    menu = DBusMenu(items)

    # Static properties (direct access works for @dbus_property).
    assert menu.Version == 3
    assert menu.TextDirection == "ltr"
    assert menu.Status == "normal"
    assert menu.IconThemePath == []

    # GetLayout at the root returns revision + full tree.
    revision, layout = _call(menu, "GetLayout", 0, -1, [])
    assert revision == 0
    root_id, _root_props, children = layout
    assert root_id == 0
    assert len(children) == 3

    # GetLayout for a non-root parent returns an empty stub.
    _rev, sub = _call(menu, "GetLayout", 1, -1, [])
    assert sub == [1, {}, []]

    # GetGroupProperties: known id returned, unknown id skipped.
    groups = _call(menu, "GetGroupProperties", [1, 999], [])
    assert len(groups) == 1
    assert groups[0][0] == 1

    # GetProperty: known prop, then unknown id -> empty fallback.
    assert _call(menu, "GetProperty", 1, "label").value == "Start Recording"
    assert _call(menu, "GetProperty", 999, "label").value == ""

    # Event: only "clicked" fires the callback.
    _call(menu, "Event", 1, "clicked", Variant("s", ""), 0)
    _call(menu, "Event", 2, "hovered", Variant("s", ""), 0)
    assert clicks == [1]

    # EventGroup: batch clicks dispatch; unknown ids reported missing.
    missing = _call(
        menu,
        "EventGroup",
        [[2, "clicked", Variant("s", ""), 0], [999, "clicked", Variant("s", ""), 0]],
    )
    assert clicks == [1, 2]
    assert missing == [999]

    # Static show hints.
    assert _call(menu, "AboutToShow", 0) is False
    assert _call(menu, "AboutToShowGroup", [1, 2]) == [[], []]

    # Signals emit cleanly without a bus.
    assert menu.LayoutUpdated() == [0, 0]
    assert menu.ItemsPropertiesUpdated() == [[], []]
    assert menu.ItemActivationRequested() == [0, 0]

    # State change: swap items -> revision bumps, host refetches a fresh layout.
    menu.replace_items(
        [MenuItem(id=1, label="Stop Recording", on_clicked=lambda: clicks.append(11))]
    )
    new_rev, new_layout = _call(menu, "GetLayout", 0, -1, [])
    assert new_rev == 1
    assert new_layout[2][0].value[0] == 1
    _call(menu, "Event", 1, "clicked", Variant("s", ""), 0)
    assert clicks == [1, 2, 11]
