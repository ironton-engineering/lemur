#!/usr/bin/python3
"""Native Lemur window via GTK + WebKit (system Python)."""
from __future__ import annotations

import sys

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import Gdk, Gtk, WebKit2  # noqa: E402


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9000"

    win = Gtk.Window(title="Lemur")
    win.set_default_size(1800, 1300)
    win.set_icon_name("utilities-terminal")
    win.connect("destroy", Gtk.main_quit)

    context = WebKit2.WebContext.get_default()
    context.set_cache_model(WebKit2.CacheModel.DOCUMENT_VIEWER)

    view = WebKit2.WebView()
    settings = view.get_settings()
    settings.set_enable_developer_extras(True)
    view.load_uri(url)

    def on_key(_widget, event: Gdk.EventKey) -> bool:
        ctrl = bool(event.state & Gdk.ModifierType.CONTROL_MASK)
        shift = bool(event.state & Gdk.ModifierType.SHIFT_MASK)
        key = event.keyval
        if key == Gdk.KEY_F5 or (ctrl and key == Gdk.KEY_r):
            if shift and ctrl:
                view.get_context().clear_cache()
            view.reload()
            return True
        return False

    win.connect("key-press-event", on_key)

    # No GTK scroll chrome — the web UI owns viewport layout.
    win.add(view)
    win.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
