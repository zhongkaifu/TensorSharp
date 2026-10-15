"""Security/transport tests using only loopback HTTP and a mocked CLI (no browser)."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import urlencode


SOURCE = Path(__file__).resolve().parents[1] / "browser-secret-handoff.py"
spec = importlib.util.spec_from_file_location("browser_secret_handoff", SOURCE)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class SecretHandoffTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.calls = []
        self.config = SimpleNamespace(
            node=Path("/absolute/node"), cli=Path("/absolute/cli.js"),
            cwd=Path("/absolute/workspace"), daemon_session_dir=Path("/absolute/daemon"),
            session="existing-session", expected_origin="https://example.test",
            expected_path="/login/", username_label="Email or username",
            expected_username='person+"quoted"@example.test', password_label="Password",
            submit_label="Log In")
        self.server = handoff.HandoffServer(self.config, run=self.run_cli, clock=lambda: self.now)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read().decode(), dict(response.getheaders())
        finally:
            connection.close()

    def secret_headers(self):
        return {"Origin": self.server.origin, "X-TensorSharp-Handoff": self.server.token}

    def post_password(self, password="synthetic-only-密&+$(do-not-expand)", **headers):
        values = {"Origin": self.server.origin, "Content-Type": "application/x-www-form-urlencoded"}
        values.update(headers)
        return self.request("POST", self.server.form_path, urlencode({"password": password}), values)

    def run_cli(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        status, secret, _ = self.request("GET", self.server.secret_path, headers=self.secret_headers())
        self.assertEqual(200, status)
        self.assertTrue(secret)
        self.assertNotIn(secret, json.dumps(argv))
        self.assertNotIn(secret, json.dumps(kwargs["env"]))
        self.assertEqual(410, self.request("GET", self.server.secret_path,
                                          headers=self.secret_headers())[0])
        self.assertIsNone(self.server.password)
        self.assertEqual(200, self.request("POST", self.server.result_path, "submitted",
                                          self.secret_headers())[0])
        return SimpleNamespace(returncode=0)

    def test_form_is_local_has_no_external_resources_and_disables_storage(self):
        self.assertEqual("127.0.0.1", self.server.server_address[0])
        self.assertGreater(self.server.server_port, 0)
        self.assertNotEqual(self.server.form_path, self.server.secret_path)
        self.assertGreaterEqual(len(self.server.form_path), 43)
        status, body, headers = self.request("GET", self.server.form_path)
        self.assertEqual(200, status)
        self.assertIn("type=password", body)
        self.assertNotIn("<script", body)
        self.assertNotIn(self.server.secret_path, body)
        self.assertNotIn(self.server.token, body)
        self.assertIn("&quot;quoted&quot;", body)
        self.assertEqual("no-store", headers["Cache-Control"])
        self.assertEqual("same-origin", headers["Referrer-Policy"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_native_form_keeps_same_origin_policy_without_accepting_null_origin(self):
        _, body, headers = self.request("GET", self.server.form_path)
        # Both policies matter: HTML's meta can override the HTTP response policy.
        # no-referrer makes native POST's Origin null, which must remain rejected.
        self.assertEqual("same-origin", headers["Referrer-Policy"])
        self.assertIn("<meta name=referrer content=same-origin>", body)
        self.assertNotIn("no-referrer", body)
        self.assertEqual(403, self.post_password(Origin="null")[0])
        self.assertFalse(self.server.attempted)
        self.assertEqual(200, self.post_password()[0])

    def test_valid_submission_transports_secret_once_without_argv_env_or_output(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            status, body, _ = self.post_password()
        self.assertEqual(200, status)
        self.assertIn("Sign-in still needs verification", body)
        self.assertEqual("", output.getvalue())
        self.assertEqual("submitted", self.server.status)
        self.assertTrue(self.server.done.wait(1))
        argv, options = self.calls[0]
        self.assertEqual([str(self.config.node), str(self.config.cli), "--session", "existing-session", "run-code"], argv[:5])
        self.assertEqual(str(self.config.cwd), options["cwd"])
        self.assertEqual(str(self.config.daemon_session_dir), options["env"]["PWTEST_DAEMON_SESSION_DIR"])
        self.assertEqual(os.environ.get("HOME"), options["env"].get("HOME"))
        self.assertEqual(subprocess.DEVNULL, options["stdout"])
        self.assertEqual(subprocess.DEVNULL, options["stderr"])
        self.assertEqual(subprocess.DEVNULL, options["stdin"])
        self.assertNotIn("shell", options)
        self.assertLessEqual(options["timeout"], 60)
        self.assertIsNone(self.server.password)

    def test_wrong_unknown_and_reused_routes_do_not_launch_cli(self):
        for path in ("/", self.server.form_path + "/", self.server.form_path + "?x=1"):
            self.assertEqual(404, self.request("GET", path)[0])
        self.assertEqual(404, self.request("POST", "/", "password=x")[0])
        self.assertEqual(200, self.post_password()[0])
        self.assertEqual(410, self.post_password()[0])
        self.assertEqual(410, self.request("GET", self.server.form_path)[0])
        self.assertEqual(1, len(self.calls))

    def test_secret_requires_distinct_capability_header_and_origin(self):
        for headers in ({}, {"Origin": self.server.origin},
                        {"Origin": "https://attacker.test", "X-TensorSharp-Handoff": self.server.token},
                        {"Origin": self.server.origin, "X-TensorSharp-Handoff": "wrong"}):
            self.assertEqual(404, self.request("GET", self.server.secret_path, headers=headers)[0])
        self.assertEqual(410, self.request("GET", self.server.secret_path,
                                          headers=self.secret_headers())[0])
        self.assertFalse(self.server.fetched)
        self.assertEqual(200, self.post_password()[0])

    def test_form_rejects_wrong_or_missing_origin_host_and_content_type(self):
        for headers in ({"Origin": "https://attacker.test"}, {"Origin": "null"},
                        {"Host": "localhost:" + str(self.server.server_port)},
                        {"Content-Type": "text/plain"}):
            self.assertEqual(403, self.post_password(**headers)[0])
        self.assertEqual(403, self.request("POST", self.server.form_path, "password=x",
                                          {"Content-Type": "application/x-www-form-urlencoded"})[0])
        self.assertFalse(self.calls)

    def test_duplicate_host_or_origin_is_rejected(self):
        for name, value in (("Host", self.server.host), ("Origin", self.server.origin)):
            connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
            try:
                connection.putrequest("POST", self.server.form_path)
                connection.putheader("Origin", self.server.origin)
                connection.putheader(name, value)
                connection.putheader("Content-Type", "application/x-www-form-urlencoded")
                connection.putheader("Content-Length", "10")
                connection.endheaders(b"password=x")
                response = connection.getresponse()
                self.assertEqual(403, response.status)
                response.read()
            finally:
                connection.close()
        self.assertFalse(self.calls)

    def test_empty_duplicate_extra_and_malformed_form_fields_are_rejected(self):
        for body in ("password=", "password=x&password=y", "password=x&other=y", "other=x", "password"):
            result = self.request("POST", self.server.form_path, body,
                                  {"Origin": self.server.origin,
                                   "Content-Type": "application/x-www-form-urlencoded"})
            self.assertEqual(400, result[0])
        self.assertFalse(self.calls)

    def test_body_size_and_transfer_encoding_are_bounded(self):
        self.assertEqual(400, self.post_password("x" * handoff.MAX_BODY)[0])
        for headers in ({"Transfer-Encoding": "chunked"}, {"Content-Length": "-1"},
                        {"Content-Length": "invalid"}, {"Content-Length": "0"}):
            self.assertEqual(400, self.post_password(**headers)[0])
        self.assertFalse(self.calls)

    def test_malformed_post_framing_returns_refusal_without_consuming_a_handoff(self):
        # Supply bytes without HTTPConnection synthesizing a Content-Length.
        # Rejection cleanup must not use any of these untrusted lengths.
        framing_cases = ([], [("Content-Length", "invalid")],
                         [("Content-Length", "0")],
                         [("Content-Length", str(handoff.MAX_BODY + 1))],
                         [("Content-Length", str(2 ** 64))],
                         [("Content-Length", "10"), ("Content-Length", "10")],
                         [("Transfer-Encoding", "chunked")])
        for framing in framing_cases:
            with self.subTest(framing=framing):
                connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                                        timeout=5)
                try:
                    connection.putrequest("POST", self.server.form_path)
                    connection.putheader("Origin", self.server.origin)
                    connection.putheader("Content-Type", "application/x-www-form-urlencoded")
                    for name, value in framing:
                        connection.putheader(name, value)
                    connection.endheaders(b"password=x")
                    response = connection.getresponse()
                    self.assertEqual(400, response.status)
                    self.assertEqual(b"Request rejected.", response.read())
                finally:
                    connection.close()
        self.assertFalse(self.server.attempted)
        self.assertIsNone(self.server.password)
        self.assertFalse(self.calls)
        self.assertEqual(200, self.post_password()[0])

    def test_refusal_is_sent_without_waiting_for_a_declared_body(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.putrequest("POST", self.server.form_path)
            connection.putheader("Origin", "https://attacker.test")
            connection.putheader("Content-Type", "application/x-www-form-urlencoded")
            connection.putheader("Content-Length", str(2 ** 64))
            connection.endheaders()  # An untrusted client never sends its promised body.
            response = connection.getresponse()
            self.assertEqual(403, response.status)
            self.assertEqual(b"Request rejected.", response.read())
        finally:
            connection.close()
        self.assertFalse(self.calls)
        self.assertFalse(self.server.attempted)

    def test_rejection_cleanup_has_independent_byte_and_time_bounds(self):
        class Input:
            bytes_read = 0
            calls = 0

            def read1(self, size):
                self.bytes_read += size
                self.calls += 1
                return b"x" * size

        handler = object.__new__(handoff.HandoffHandler)
        handler.connection = SimpleNamespace(settimeout=lambda timeout: None)
        handler.rfile = Input()
        with patch.object(handoff.time, "monotonic", return_value=100.0):
            handler.discard_rejected_body()
        self.assertEqual(handoff.MAX_REJECT_DRAIN, handler.rfile.bytes_read)
        self.assertGreater(handler.rfile.calls, 0)

        handler.rfile = Input()
        with patch.object(handoff.time, "monotonic", side_effect=(
                100.0, 100.0, 100.0 + handoff.REJECT_DRAIN_SECONDS)):
            handler.discard_rejected_body()
        self.assertEqual(1, handler.rfile.calls)
        self.assertLess(handler.rfile.bytes_read, handoff.MAX_REJECT_DRAIN)

    def test_concurrent_posts_launch_exactly_one_cli_and_fetch(self):
        entered, release = threading.Event(), threading.Event()
        def slow_run(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return self.run_cli(*args, **kwargs)
        self.server.run = slow_run
        with ThreadPoolExecutor(max_workers=8) as pool:
            first = pool.submit(self.post_password)
            self.assertTrue(entered.wait(2))
            other = list(pool.map(lambda _: self.post_password()[0], range(6)))
            release.set()
            self.assertEqual(200, first.result()[0])
        self.assertEqual([410] * 6, other)
        self.assertEqual(1, len(self.calls))

    def test_expiry_rejects_form_post_and_clears_secret(self):
        self.server.password = "must-be-cleared"
        self.now += handoff.LIFETIME_SECONDS
        self.assertEqual(410, self.request("GET", self.server.form_path)[0])
        self.assertEqual(410, self.post_password()[0])
        self.assertEqual(410, self.request("GET", self.server.secret_path,
                                          headers=self.secret_headers())[0])
        self.assertIsNone(self.server.password)
        self.assertFalse(self.calls)

    def test_failure_or_timeout_has_no_secret_output_and_no_retry(self):
        secret = "synthetic-sensitive-failure-value"
        def fail(*args, **kwargs):
            raise subprocess.TimeoutExpired(secret, 1, output=secret, stderr=secret)
        self.server.run = fail
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            status, body, _ = self.post_password(secret)
        self.assertEqual(502, status)
        self.assertNotIn(secret, body + output.getvalue())
        self.assertIsNone(self.server.password)
        self.assertEqual(410, self.post_password(secret)[0])

    def test_zero_exit_without_submission_receipt_is_failure(self):
        self.server.run = lambda *args, **kwargs: SimpleNamespace(returncode=0)
        self.assertEqual(502, self.post_password()[0])
        self.assertIsNone(self.server.password)

    def test_submission_receipt_requires_fetch_and_is_single_use(self):
        self.server.attempted = True
        self.assertEqual(409, self.request("POST", self.server.result_path, "submitted",
                                          self.secret_headers())[0])
        self.server.fetched = True
        self.assertEqual(200, self.request("POST", self.server.result_path, "submitted",
                                          self.secret_headers())[0])
        self.assertEqual(409, self.request("POST", self.server.result_path, "submitted",
                                          self.secret_headers())[0])

    def test_script_encodes_metadata_and_guards_before_fetch_fill_and_click(self):
        code = handoff.build_code(self.config, self.server.origin + self.server.secret_path,
                                  self.server.origin + self.server.result_path, self.server.token)
        self.assertIn(json.dumps(self.config.expected_username), code)
        first_guard = code.index("await guard();")
        fetch = code.index("await page.request.get")
        second_guard = code.index("await guard();", first_guard + 1)
        fill = code.index("await passwordField.fill(password")
        third_guard = code.index("await guard();", second_guard + 1)
        click = code.index("await submit.click")
        self.assertLess(first_guard, fetch)
        self.assertLess(fetch, second_guard)
        self.assertLess(second_guard, fill)
        self.assertLess(fill, third_guard)
        self.assertLess(third_guard, click)
        self.assertIn("await response.dispose()", code)
        self.assertIn("maxRedirects: 0", code)
        self.assertNotIn("snapshot", code)
        self.assertNotIn("console.", code)
        self.assertNotIn("new URL", code)

    def execute_script(self, scenario):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is unavailable for the run-code VM behavior test")
        code = handoff.build_code(self.config, self.server.origin + self.server.secret_path,
                                  self.server.origin + self.server.result_path, self.server.token)
        # Input/code travel over stdin only. The fake password is never a CLI arg.
        program = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const {scenario: s, username, origin, path} = input;
const actions = [];
let address = {origin, path}, account = username, enabled = false;
const usernameField = {isVisible: async()=>true, inputValue: async()=>account};
const passwordField = {
  isVisible: async()=>true, isEditable: async()=>true,
  getAttribute: async()=>s.wrongType ? 'text' : 'password',
  fill: async(value)=> {
    if (value && value !== 'FAKE_PASSWORD') throw new Error('Unexpected value');
    actions.push(value ? 'fill' : 'clear');
    if (value && s.navigateAfterFill) address = {origin:'https://other.test', path:'/'};
  }
};
const submit = {
  isVisible: async()=>true,
  waitForElementState: async(state)=> { actions.push('wait-' + state); enabled = true; },
  click: async()=> { if (!enabled) throw new Error('Disabled'); actions.push('click'); }
};
const locator = (handle, count)=>({count:async()=>count, elementHandle:async()=>handle});
const response = ()=>({status:()=>200, dispose:async()=>actions.push('dispose')});
const request = {
  tracing:{_isTracing:false},
  get:async()=> {
    actions.push('fetch');
    if (s.changeAccountAfterFetch) account='different-account';
    if (s.navigateAfterFetch) address={origin:'https://other.test',path:'/'};
    return {...response(), text:async()=>'FAKE_PASSWORD'};
  },
  post:async(url, options)=> { actions.push('report-' + options.data); return response(); }
};
const page = {
  request,
  context:()=>({_options:s.proxy ? {proxy:{server:'http://proxy'}} : {},
    tracing:{_isTracing:!!s.recording}, request, browser:()=>({_options:{}})}),
  evaluate:async()=>address,
  getByLabel:(name, options)=> {
    if (!options.exact) throw new Error('Must be exact');
    return locator(name==='Password' ? passwordField : usernameField, s.labelCount ?? 1);
  },
  getByRole:(role, options)=> {
    if (!options.exact) throw new Error('Must be exact');
    if (role==='button') return locator(submit, 1);
    if (role!=='textbox') throw new Error('Unexpected role');
    actions.push('role-' + options.name);
    return locator(options.name==='Password' ? passwordField : usernameField, 1);
  }
};
vm.runInNewContext('(' + input.code + ')(page)', {page}).then(
  result=>process.stdout.write(JSON.stringify({result, actions})),
  ()=>process.exit(1));
"""
        result = subprocess.run([node, "-e", program], input=json.dumps({
            "code": code, "scenario": scenario, "username": self.config.expected_username,
            "origin": self.config.expected_origin, "path": self.config.expected_path}),
            capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual("", result.stderr)
        self.assertNotIn("FAKE_PASSWORD", result.stdout)
        return json.loads(result.stdout)

    def test_script_fills_role_fallback_then_waits_for_disabled_submit(self):
        result = self.execute_script({"labelCount": 0})
        self.assertEqual("submitted", result["result"])
        actions = result["actions"]
        self.assertIn("role-Email or username", actions)
        self.assertIn("role-Password", actions)
        self.assertLess(actions.index("fetch"), actions.index("fill"))
        self.assertLess(actions.index("dispose"), actions.index("fill"))
        self.assertLess(actions.index("fill"), actions.index("wait-enabled"))
        self.assertLess(actions.index("wait-enabled"), actions.index("click"))
        self.assertIn("report-submitted", actions)

    def test_script_rejects_ambiguous_fields_recording_proxy_and_wrong_type_before_fetch(self):
        for scenario in ({"labelCount": 2}, {"recording": True}, {"proxy": True}, {"wrongType": True}):
            with self.subTest(scenario=scenario):
                result = self.execute_script(scenario)
                self.assertEqual("failed", result["result"])
                self.assertNotIn("fetch", result["actions"])
                self.assertNotIn("fill", result["actions"])

    def test_script_rechecks_navigation_and_account_after_fetch_before_filling(self):
        for scenario in ({"changeAccountAfterFetch": True}, {"navigateAfterFetch": True}):
            with self.subTest(scenario=scenario):
                result = self.execute_script(scenario)
                self.assertEqual("failed", result["result"])
                self.assertIn("fetch", result["actions"])
                self.assertIn("dispose", result["actions"])
                self.assertNotIn("fill", result["actions"])
                self.assertNotIn("click", result["actions"])

    def test_script_rechecks_navigation_before_click_and_clears_failed_input(self):
        result = self.execute_script({"navigateAfterFill": True})
        self.assertEqual("failed", result["result"])
        self.assertIn("fill", result["actions"])
        self.assertIn("clear", result["actions"])
        self.assertNotIn("click", result["actions"])


if __name__ == "__main__":
    unittest.main()
