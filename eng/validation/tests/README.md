# Validation test environments

Run a selected suite from the repository root, for example:

```text
python -m unittest discover -s eng/validation/tests -p test_deepseek_teacher_capture.py
python -m unittest discover -s eng/validation/tests -p test_deepseek_teacher_baseline.py
python -m unittest discover -s eng/validation/tests -p test_browser_secret_handoff.py
```

The two teacher suites above use fake native APIs and small synthetic rows; they
do not load a model or qualify model quality. The browser handoff suite uses
loopback HTTP, a mocked CLI, and an optional Node VM test, without a browser.
Path assertions use the configured native `Path` representation, including
Windows separators, while preserving the exact executable and argument values.

`capture-deepseek-teacher-logits.py` verifies the exact SHA-256 of
`compare-deepseek-teacher-logits.py` before importing it. The comparator's
`.gitattributes` entry fixes LF bytes on every checkout platform. This preserves
the existing audited hash; it does not normalize arbitrary evidence during hash
verification or relax the capture's identity checks. Existing checkouts with
CRLF need this one file refreshed to the indexed LF content before these suites
can run. Preserve local edits before refreshing a checkout file.

Other suites have separate prerequisites. In particular, historical DeepSeek
body/profile tests require the original pinned engine source and archived
reports under ignored `docs/validation/`; today's engine source is not a valid
substitute. Missing archives must remain an unavailable historical validation,
not regenerated successful model evidence. Required reusable fixtures belong
in the test project and generated reports remain ignored.

The Playwright shell-wrapper tests require a coherent POSIX Python/bash/PATH
environment for their fake `npx` executable. On Windows, `bash.exe` may be the
WSL launcher; passing native Windows paths and a Windows Python shebang across
that boundary does not establish a working fixture. Run those tests inside an
explicitly prepared compatible shell environment. A skipped or unavailable
platform scenario never counts as a passing model or device validation.
