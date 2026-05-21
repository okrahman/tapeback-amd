"""DBusMenu layout / property building — pure helpers, no D-Bus required."""

from tests.fixtures import requires_dbus_next

pytestmark = requires_dbus_next

from tapeback._dbusmenu import MenuItem, _item_properties, build_layout  # noqa: E402


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
