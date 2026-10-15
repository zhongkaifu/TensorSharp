#!/usr/bin/env python3
"""One-use local password entry for an already open, inspected browser login form.

The password stays out of chat, CLI arguments, helper logs and artifacts. Run only
against a trusted local CLI/browser with recording/debug capture disabled. Browser,
website and OS memory handling remain outside this helper's control. This helper
submits once; a separate browser observation must establish whether login worked.
"""

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit


MAX_BODY = 16384
MAX_REJECT_DRAIN = 2 * MAX_BODY
REJECT_DRAIN_SECONDS = 0.25
LIFETIME_SECONDS = 600


def build_code(config, secret_url, result_url, token):
    """Build code containing capabilities and observed form metadata, never a password."""
    values = json.dumps({
        "origin": config.expected_origin, "path": config.expected_path,
        "usernameLabel": config.username_label, "username": config.expected_username,
        "passwordLabel": config.password_label, "submitLabel": config.submit_label,
        "secretUrl": secret_url, "resultUrl": result_url, "token": token,
        "localOrigin": secret_url.rsplit("/", 1)[0],
    }, ensure_ascii=True)
    # run-code executes in a VM exposing `page`, but not Node's URL/fetch globals.
    return """async (page) => {
  const c = CONFIG;
  let password = '', passwordField, filled = false;
  const headers = {'X-TensorSharp-Handoff': c.token, 'Origin': c.localOrigin};
  const report = async (status) => {
    const r = await page.request.post(c.resultUrl,
      {headers, data: status, timeout: 5000, maxRedirects: 0});
    try { if (r.status() !== 200) throw new Error('Handoff failed'); }
    finally { await r.dispose(); }
  };
  try {
    const context = page.context();
    // The caller must use a known local unrecorded session. Reject explicit
    // recording/proxy settings where the current Playwright client exposes them.
    const browser = context.browser();
    const recording = (tracing) => tracing?._isTracing || tracing?._harRecorders?.size;
    if (recording(context.tracing) || recording(context.request?.tracing) ||
        context._options?.recordHar || context._options?.recordVideo ||
        context._options?.proxy || browser?._options?.proxy || context._logger || browser?._logger)
      throw new Error('Handoff unavailable');
    const locate = async (locator) => {
      if (await locator.count() !== 1) throw new Error('Unexpected form');
      const handle = await locator.elementHandle();
      if (!handle || !await handle.isVisible()) throw new Error('Unexpected form');
      return handle;
    };
    const field = async (label) => {
      const byLabel = page.getByLabel(label, {exact: true});
      return locate(await byLabel.count() === 0
        ? page.getByRole('textbox', {name: label, exact: true}) : byLabel);
    };
    const usernameField = await field(c.usernameLabel);
    passwordField = await field(c.passwordLabel);
    const submit = await locate(page.getByRole('button', {name: c.submitLabel, exact: true}));
    const guard = async () => {
      const address = await page.evaluate(() => ({origin: location.origin, path: location.pathname}));
      if (address.origin !== c.origin || address.path !== c.path ||
          await usernameField.inputValue() !== c.username ||
          await passwordField.getAttribute('type') !== 'password' ||
          !await passwordField.isEditable())
        throw new Error('Unexpected form');
    };
    await guard();
    const response = await page.request.get(c.secretUrl,
      {headers, timeout: 5000, maxRedirects: 0});
    try {
      if (response.status() !== 200) throw new Error('Handoff unavailable');
      password = await response.text();
    } finally { await response.dispose(); }
    await guard();
    await passwordField.fill(password, {timeout: 5000});
    filled = true;
    password = '';
    await submit.waitForElementState('enabled', {timeout: 5000});
    await guard();
    await submit.click({timeout: 10000});
    await report('submitted');
    return 'submitted';
  } catch {
    password = '';
    if (filled && passwordField) {
      try { await passwordField.fill('', {timeout: 1000}); } catch {}
    }
    try { await report('failed'); } catch {}
    return 'failed';
  }
}""".replace("CONFIG", values, 1)


class HandoffServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config, *, run=subprocess.run, clock=time.monotonic):
        super().__init__(("127.0.0.1", 0), HandoffHandler)
        self.config, self.run, self.clock = config, run, clock
        self.timeout = 0.5
        self.lock = threading.Lock()
        self.expires = clock() + LIFETIME_SECONDS
        self.form_path = "/" + secrets.token_urlsafe(32)
        self.secret_path = "/" + secrets.token_urlsafe(32)
        self.result_path = self.secret_path + "/result"
        self.token = secrets.token_urlsafe(32)
        self.origin = "http://127.0.0.1:" + str(self.server_port)
        self.host = "127.0.0.1:" + str(self.server_port)
        self.url = self.origin + self.form_path
        self.password = None
        self.attempted = False
        self.fetched = False
        self.receipt = None
        self.status = "waiting"
        self.done = threading.Event()

    def expired(self):
        if self.clock() >= self.expires:
            self.password = None
            return True
        return False

    def handle_error(self, request, client_address):
        # BaseServer would print tracebacks; request bodies must never be logged.
        pass

    def invoke(self):
        code = build_code(self.config, self.origin + self.secret_path,
                          self.origin + self.result_path, self.token)
        env = os.environ.copy()
        env["PWTEST_DAEMON_SESSION_DIR"] = str(self.config.daemon_session_dir)
        try:
            result = self.run([str(self.config.node), str(self.config.cli),
                               "--session", self.config.session, "run-code", code],
                              cwd=str(self.config.cwd), env=env, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=min(60, max(0.1, self.expires - self.clock())),
                              check=False)
            with self.lock:
                return result.returncode == 0 and self.receipt == "submitted"
        except Exception:
            return False
        finally:
            with self.lock:
                self.password = None


class HandoffHandler(BaseHTTPRequestHandler):
    server_version = "LocalHandoff"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        self.respond(code, "Request rejected.")

    def respond(self, code, body, content_type="text/plain; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        for key, value in {
            "Content-Type": content_type, "Content-Length": str(len(data)),
            "Cache-Control": "no-store", "Pragma": "no-cache",
            # Fetch's Origin-header algorithm makes a native form POST's Origin
            # null under no-referrer. same-origin preserves our strict local
            # Origin check without sending capability URLs to other origins.
            # https://fetch.spec.whatwg.org/#append-a-request-origin-header
            "Referrer-Policy": "same-origin", "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            "Cross-Origin-Resource-Policy": "same-origin", "Connection": "close",
        }.items():
            self.send_header(key, value)
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(data)
            if code >= 400:
                # Closing with an unread POST body can reset the connection on
                # Windows, losing even the refusal status. Send the complete
                # refusal and a FIN first, then discard a bounded amount of
                # untrusted input. Never trust its framing or decode/log it.
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.discard_rejected_body()
        except (OSError, TimeoutError):
            pass

    def discard_rejected_body(self):
        remaining = MAX_REJECT_DRAIN
        deadline = time.monotonic() + REJECT_DRAIN_SECONDS
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                return
            self.connection.settimeout(timeout)
            data = self.rfile.read1(min(4096, remaining))
            if not data:
                return
            remaining -= len(data)

    def check_host(self):
        return self.headers.get_all("Host", []) == [self.server.host]

    def check_secret(self):
        return (self.check_host() and
                self.headers.get_all("Origin", []) == [self.server.origin] and
                self.headers.get_all("X-TensorSharp-Handoff", []) == [self.server.token])

    def read_body(self, limit):
        sizes = self.headers.get_all("Content-Length", [])
        if (self.headers.get_all("Transfer-Encoding") or len(sizes) != 1 or
                not sizes[0].isascii() or not sizes[0].isdigit()):
            raise ValueError()
        size = int(sizes[0])
        if not 0 < size <= limit:
            raise ValueError()
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError()
        return data.decode("utf-8", errors="strict")

    def do_GET(self):
        server = self.server
        if not self.check_host():
            self.respond(403, "Request rejected.")
            return
        with server.lock:
            if server.expired():
                self.respond(410, "This handoff has expired.")
                return
            if self.path == server.secret_path and self.check_secret():
                if server.password is None or server.fetched:
                    self.respond(410, "Password unavailable.")
                    return
                password, server.password = server.password, None
                server.fetched = True
                self.respond(200, password)
                password = None
                return
            if self.path != server.form_path:
                self.respond(404, "Not found.")
                return
            if server.attempted:
                self.respond(410, "This handoff has already been used.")
                return
        self.respond(200, "<!doctype html><meta charset=utf-8><meta name=referrer content=same-origin>"
                     "<title>One-use browser password entry</title><h1>Enter password</h1>"
                     "<p>Destination: " + html.escape(server.config.expected_origin) + "</p>"
                     "<p>Account: " + html.escape(server.config.expected_username) + "</p>"
                     "<p>This submits the inspected login form once. It does not confirm sign-in.</p>"
                     "<form method=post autocomplete=off action=\"" + server.form_path + "\">"
                     "<label>Password <input type=password name=password autocomplete=off "
                     "required autofocus></label> <button type=submit>Fill and submit</button></form>",
                     "text/html; charset=utf-8")

    def do_POST(self):
        server = self.server
        if not self.check_host():
            self.respond(403, "Request rejected.")
            return
        if self.path == server.result_path:
            if not self.check_secret():
                self.respond(403, "Request rejected.")
                return
            try:
                status = self.read_body(16)
            except (ValueError, OSError):
                self.respond(400, "Request rejected.")
                return
            with server.lock:
                if (server.expired() or not server.attempted or server.receipt is not None or
                        status not in ("submitted", "failed") or
                        (status == "submitted" and not server.fetched)):
                    self.respond(409, "Request rejected.")
                    return
                server.receipt = status
            self.respond(200, "Recorded.")
            return
        if self.path != server.form_path:
            self.respond(404, "Not found.")
            return
        if (self.headers.get_all("Origin", []) != [server.origin] or
                self.headers.get_all("Content-Type", []) != ["application/x-www-form-urlencoded"]):
            self.respond(403, "Request rejected.")
            return
        try:
            body = self.read_body(MAX_BODY)
            fields = parse_qs(body, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
            body = None
            if set(fields) != {"password"} or len(fields["password"]) != 1 or not fields["password"][0]:
                raise ValueError()
        except (ValueError, OSError):
            self.respond(400, "Request rejected.")
            return
        with server.lock:
            if server.expired() or server.attempted:
                self.respond(410, "This handoff is unavailable.")
                return
            server.password = fields["password"].pop()
            fields = None
            server.attempted = True
            server.status = "submitting"
        submitted = server.invoke()
        server.status = "submitted" if submitted else "failed"
        self.respond(200 if submitted else 502,
                     "Login form submitted. Sign-in still needs verification." if submitted else
                     "Submission could not be confirmed. Do not retry here; return to your assistant.")
        server.done.set()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("node", "cli", "cwd", "daemon-session-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("session", "expected-origin", "expected-path", "username-label",
                 "expected-username", "password-label", "submit-label"):
        parser.add_argument("--" + flag, required=True)
    config = parser.parse_args()
    for name in ("node", "cli", "cwd", "daemon_session_dir"):
        path = getattr(config, name)
        if not path.is_absolute() or not path.exists():
            parser.error(name + " must be an existing absolute path")
    origin = urlsplit(config.expected_origin)
    if (origin.scheme not in ("http", "https") or not origin.hostname or origin.username or
            origin.password or origin.path or origin.query or origin.fragment or
            not config.expected_path.startswith("/") or "?" in config.expected_path or
            "#" in config.expected_path):
        parser.error("expected-origin must be an origin and expected-path a pathname")
    return config


def main():
    server = HandoffServer(parse_args())
    print(server.url, flush=True)
    try:
        while not server.done.is_set() and server.clock() < server.expires:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        with server.lock:
            server.password = None
        server.server_close()
    print(server.status if server.done.is_set() else "expired", flush=True)


if __name__ == "__main__":
    main()
