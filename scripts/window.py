#!/usr/bin/python3
"""Native Lemur window via GTK + WebKit (system Python)."""
from __future__ import annotations

import sys
from pathlib import Path

import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
try:
    gi.require_version("WebKit2", "4.1")
except ValueError:
    gi.require_version("WebKit2", "4.0")
from gi.repository import Gdk, Gtk, WebKit2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ICON_FILE = ROOT / "assets" / "lemur.png"
APPLICATION_ID = "io.github.ironton_engineering.Lemur"


def _apply_window_icon(window: Gtk.Window) -> None:
    if ICON_FILE.is_file():
        window.set_icon_from_file(str(ICON_FILE))
    window.set_icon_name("lemur")


class LemurApplication(Gtk.Application):
    def __init__(self, url: str) -> None:
        super().__init__(application_id=APPLICATION_ID)
        self.url = url

    def do_activate(self) -> None:
        win = self.get_active_window()
        if win is not None:
            win.present()
            return

        win = Gtk.ApplicationWindow(application=self, title="Lemur")
        win.set_default_size(1800, 1300)
        _apply_window_icon(win)

        context = WebKit2.WebContext.get_default()
        context.set_cache_model(WebKit2.CacheModel.DOCUMENT_VIEWER)

        view = WebKit2.WebView()
        view.load_uri(self.url)

        # No GTK scroll chrome — the web UI owns viewport layout.
        win.add(view)
        win.show_all()


def main() -> int:
    Gdk.set_program_class("Lemur")
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9000"
    app = LemurApplication(url)
    return app.run([sys.argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
