"""UI server guards: Host-header check (DNS rebinding) and torchrun port selection."""
import socket

import pytest

from ui.app import _allowed_host
from ui.queue import find_free_port


@pytest.mark.parametrize("host", [
    "localhost", "localhost:8787", "127.0.0.1:8787", "[::1]:8787",
    "10.0.0.5:8787",                       # remote access by IP keeps working
    socket.gethostname(),
])
def test_allowed_hosts(host):
    assert _allowed_host(host)


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:8787", "", "evil.localhost.example"])
def test_rebinding_hosts_rejected(host):
    assert not _allowed_host(host)


def test_extra_hosts_from_env(monkeypatch):
    monkeypatch.setenv("OMNI_UI_ALLOWED_HOSTS", "gpu-box.example.org, other.lan")
    assert _allowed_host("gpu-box.example.org:8787")
    assert _allowed_host("other.lan")
    assert not _allowed_host("attacker.example")


def test_find_free_port_is_bindable():
    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", port))
