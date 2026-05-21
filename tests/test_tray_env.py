"""Tray environment detection — picks the right cases for the AppIndicator hint."""

from tapeback._tray_env import detect_tray_env


def test_x11_no_hint():
    env = detect_tray_env({"XDG_SESSION_TYPE": "x11", "XDG_CURRENT_DESKTOP": "GNOME"})
    assert env.needs_appindicator_hint is False
    assert env.hint_message == ""


def test_wayland_gnome_needs_hint():
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"})
    assert env.needs_appindicator_hint is True
    assert "gnome-shell-extension-appindicator" in env.hint_message


def test_wayland_ubuntu_gnome_needs_hint():
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "ubuntu:GNOME"})
    assert env.needs_appindicator_hint is True


def test_wayland_plasma_no_hint():
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "KDE"})
    assert env.needs_appindicator_hint is False
    assert env.hint_message == ""


def test_missing_envvars_no_hint():
    env = detect_tray_env({})
    assert env.needs_appindicator_hint is False
    assert env.session_type == ""
    assert env.desktop == ""


def test_hint_message_contains_install_command():
    env = detect_tray_env({"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"})
    assert "sudo apt install gnome-shell-extension-appindicator" in env.hint_message
    assert "tapeback start" in env.hint_message
