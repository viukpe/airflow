#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import contextlib
import socket
import threading
from unittest import mock

import paramiko
import pytest

from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.ssh.tunnel import SSHTunnel


@pytest.fixture
def mock_ssh_client():
    client = mock.MagicMock(spec=paramiko.SSHClient)
    transport = mock.MagicMock(spec=paramiko.Transport)
    transport.is_active.return_value = True
    client.get_transport.return_value = transport
    return client


class _EchoSSHServerInterface(paramiko.ServerInterface):
    """Accepts direct-tcpip channels (what port-forwarding opens) and auth."""

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED

    def check_channel_direct_tcpip_request(self, chanid, origin, destination):
        return paramiko.OPEN_SUCCEEDED

    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL

    def get_allowed_auths(self, username):
        return "password"


class _RealEchoSSHServer:
    """A minimal real paramiko SSH server that echoes bytes on forwarded channels."""

    def __init__(self):
        self._host_key = paramiko.RSAKey.generate(2048)
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("localhost", 0))
        self._listener.listen(50)
        self.host, self.port = self._listener.getsockname()
        self._running = False

    @staticmethod
    def _echo(chan):
        try:
            while True:
                data = chan.recv(16384)
                if not data:
                    break
                chan.sendall(data)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                chan.close()

    def _serve(self, client_sock):
        transport = paramiko.Transport(client_sock)
        transport.add_server_key(self._host_key)
        try:
            transport.start_server(server=_EchoSSHServerInterface())
        except Exception:
            return
        while transport.is_active():
            chan = transport.accept(timeout=5)
            if chan is None:
                continue
            threading.Thread(target=self._echo, args=(chan,), daemon=True).start()

    def _accept_loop(self):
        while self._running:
            try:
                self._listener.settimeout(0.5)
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def __enter__(self):
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._running = False
        with contextlib.suppress(OSError):
            self._listener.close()


class TestSSHTunnel:
    def test_local_bind_port_is_available_after_init(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            port = tunnel.local_bind_port
            assert isinstance(port, int)
            assert port > 0
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_local_bind_address(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            host, port = tunnel.local_bind_address
            assert host == "localhost"
            assert port == tunnel.local_bind_port
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_explicit_local_port(self, mock_ssh_client):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432, local_port=free_port)
        try:
            assert tunnel.local_bind_port == free_port
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_context_manager_starts_and_stops(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        with tunnel as t:
            assert t is tunnel
            assert tunnel._running is True
            assert tunnel._thread is not None
            assert tunnel._thread.is_alive()
        assert tunnel._running is False
        assert tunnel._thread is None

    def test_start_emits_deprecation_warning(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            with pytest.warns(AirflowProviderDeprecationWarning, match="SSHTunnel.start"):
                tunnel.start()
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_stop_emits_deprecation_warning(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        tunnel._start_forwarding()
        with pytest.warns(AirflowProviderDeprecationWarning, match="SSHTunnel.stop"):
            tunnel.stop()

    def test_getattr_migration_error(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            with pytest.raises(AttributeError, match="SSHTunnelForwarder has been replaced"):
                tunnel.tunnel_is_up
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_getattr_unknown_attribute(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
                tunnel.nonexistent
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_double_start_is_noop(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            tunnel._start_forwarding()
            thread1 = tunnel._thread
            tunnel._start_forwarding()
            assert tunnel._thread is thread1
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_stop_without_start_is_noop(self, mock_ssh_client):
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        try:
            tunnel._stop_forwarding()
        finally:
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()

    def test_forwarding_thread_accepts_and_forwards(self, mock_ssh_client):
        """Test that data is forwarded between local socket and SSH channel."""
        channel = mock.MagicMock(spec=paramiko.Channel)
        channel.closed = False
        channel.recv.return_value = b"response"

        transport = mock_ssh_client.get_transport.return_value
        transport.open_channel.return_value = channel

        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
        with tunnel:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client.connect(("localhost", tunnel.local_bind_port))
                client.sendall(b"hello")
                # Give the forwarding thread time to process
                threading.Event().wait(0.2)
            finally:
                client.close()

    def test_forwarding_works_with_high_fd_numbers(self, mock_ssh_client):
        """
        Regression test: forwarding must keep working when the process already
        holds >= 1024 open file descriptors.

        The previous select.select() loop in _serve_forever raised
        "filedescriptor out of range in select()" (caught, then the thread
        exited) once a watched fd was numbered >= FD_SETSIZE (1024), leaving a
        local port that accepts connections but never forwards them. The
        selectors (epoll/poll) based loop has no such ceiling.
        """
        channel = mock.MagicMock(spec=paramiko.Channel)
        channel.closed = False
        channel.recv.return_value = b"response"

        transport = mock_ssh_client.get_transport.return_value
        transport.open_channel.return_value = channel

        # Occupy a large block of low fd numbers so the tunnel's sockets (and
        # the accepted client socket) are assigned fd numbers >= 1024.
        padding = []
        try:
            for _ in range(1100):
                padding.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))

            tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432)
            # The listening socket should now sit above the select() ceiling.
            assert tunnel._server_socket is not None
            with tunnel:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    client.connect(("localhost", tunnel.local_bind_port))
                    client.sendall(b"hello")
                    # Give the forwarding thread time to accept + forward. With
                    # the old select() loop the thread would have died on a
                    # ValueError and this data would never reach the channel.
                    threading.Event().wait(0.3)
                    # The channel received the forwarded bytes -> loop is alive.
                    channel.sendall.assert_any_call(b"hello")
                finally:
                    client.close()
        finally:
            for sock in padding:
                sock.close()

    def test_end_to_end_forwarding_concurrent_high_fd(self):
        """
        End-to-end regression test over a real SSH server, under a high fd count.

        This exercises the full forwarding loop with multiple concurrent and
        sequential connections (each a unique payload) while the process holds
        >= 1024 open descriptors. It guards against two failure modes at once:
          * the FD_SETSIZE ceiling (old select() broke all forwarding), and
          * mishandling of the per-connection channel registration when several
            channels are watched together.
        """
        padding = []
        try:
            for _ in range(1100):
                padding.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))

            with _RealEchoSSHServer() as server:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    server.host,
                    port=server.port,
                    username="airflow",
                    password="airflow",
                    allow_agent=False,
                    look_for_keys=False,
                )

                # Forward to a plain TCP echo endpoint (the tunnel's "remote").
                echo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                echo.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                echo.bind(("localhost", 0))
                echo.listen(50)
                echo_port = echo.getsockname()[1]
                echo_running = {"on": True}

                def echo_accept():
                    while echo_running["on"]:
                        try:
                            echo.settimeout(0.5)
                            conn, _ = echo.accept()
                        except TimeoutError:
                            continue
                        except OSError:
                            break

                        def handle(c):
                            try:
                                while True:
                                    d = c.recv(16384)
                                    if not d:
                                        break
                                    c.sendall(d)
                            except OSError:
                                pass
                            finally:
                                c.close()

                        threading.Thread(target=handle, args=(conn,), daemon=True).start()

                threading.Thread(target=echo_accept, daemon=True).start()

                try:
                    tunnel = SSHTunnel(client, "localhost", echo_port)
                    with tunnel:
                        lport = tunnel.local_bind_port
                        assert tunnel._server_socket is not None

                        def roundtrip(payload):
                            c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            c.settimeout(5.0)
                            try:
                                c.connect(("localhost", lport))
                                c.sendall(payload)
                                got = b""
                                while len(got) < len(payload):
                                    chunk = c.recv(4096)
                                    if not chunk:
                                        break
                                    got += chunk
                                return got
                            finally:
                                c.close()

                        # Concurrent connections, each a unique payload.
                        results = {}

                        def worker(i):
                            results[i] = roundtrip(f"conn-{i:03d}".encode())

                        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
                        for t in threads:
                            t.start()
                        for t in threads:
                            t.join()
                        for i in range(8):
                            assert results[i] == f"conn-{i:03d}".encode(), (
                                f"connection {i} got {results[i]!r}"
                            )

                        # Sequential connections after the first batch closed
                        # (exercises forwarding staying healthy as channels go).
                        for i in range(5):
                            payload = f"seq-{i}".encode()
                            assert roundtrip(payload) == payload
                finally:
                    echo_running["on"] = False
                    with contextlib.suppress(OSError):
                        echo.close()
                    client.close()
        finally:
            for sock in padding:
                sock.close()

    def test_custom_logger(self, mock_ssh_client):
        custom_logger = mock.MagicMock()
        tunnel = SSHTunnel(mock_ssh_client, "remotehost", 5432, logger=custom_logger)
        try:
            assert tunnel._logger is custom_logger
        finally:
            tunnel._stop_forwarding()
            if tunnel._server_socket is not None:
                tunnel._server_socket.close()
