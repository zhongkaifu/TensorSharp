// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Python;
using TensorAgent.Core.Sandbox;
using TensorSharp.AgentHost.CodeExec;

namespace TensorAgent.Tests;

/// <summary>
/// Marks a test that needs a real CPython to say anything.
///
/// <para>
/// The build machine has none — the interpreter this component loads is staged
/// into an iOS app bundle — so these are skipped with the command that enables
/// them rather than passing vacuously. A test that returns early when the thing
/// it tests is absent is a test that reports success for an untested feature,
/// which is worse than no test.
/// </para>
/// </summary>
public sealed class LivePythonFactAttribute : FactAttribute
{
    /// <summary>The environment variable that points at a staged runtime or a CPython prefix.</summary>
    public const string RootVariable = "TENSORAGENT_PYTHON_ROOT";

    public LivePythonFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(RootVariable)))
        {
            Skip = $"needs an embedded CPython: set {RootVariable}=<TensorAgent/python-runtime/simulator "
                + "or a CPython prefix with lib/python3.13 and libpython3.13.dylib> and re-run";
        }
    }
}

/// <summary>
/// Marks a test that needs the runtime <c>prepare-python.sh</c> STAGES, rather than
/// any CPython 3.13.
///
/// <para>
/// The distinction is not pedantry, and getting it wrong costs a red suite that says
/// nothing about the product: a machine that points
/// <c>TENSORAGENT_PYTHON_ROOT</c> at a Homebrew prefix satisfies
/// <see cref="LivePythonFactAttribute"/> and then fails four tests whose whole subject
/// is what the STAGING put there — reportlab, openpyxl, pypdf, Pillow, defusedxml and
/// the certificate bundle. That is the machine not being ready, and it should say so
/// in the skip rather than in an assertion about a PDF.
/// </para>
/// </summary>
public sealed class LiveStagedPythonFactAttribute : FactAttribute
{
    /// <summary>What the staged runtime carries and a bare interpreter does not.</summary>
    private static readonly string[] Staged =
        ["reportlab", "openpyxl", "pypdf", "PIL", "defusedxml", "certifi", "lxml", "pptx", "docx"];

    public LiveStagedPythonFactAttribute()
    {
        string? root = Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable);
        if (string.IsNullOrWhiteSpace(root))
        {
            Skip = $"needs the staged runtime: set {LivePythonFactAttribute.RootVariable}="
                + "<TensorAgent/python-runtime/simulator or another prefix produced by "
                + "TensorAgent/scripts/prepare-python.sh> and re-run";
            return;
        }

        var options = new EnumerationOptions
        {
            RecurseSubdirectories = true,
            MaxRecursionDepth = 6,
            IgnoreInaccessible = true,
        };
        string[] missing = Staged.Where(name =>
        {
            try { return !Directory.EnumerateDirectories(root, name, options).Any(); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { return true; }
        }).ToArray();

        if (missing.Length > 0)
        {
            Skip = $"{root} is a CPython, not a runtime staged by prepare-python.sh: "
                + string.Join(", ", missing) + " are not under it, and these tests are about "
                + "exactly what the staging puts there";
        }
    }
}

/// <summary>
/// What can be tested about the embedded interpreter on a machine that does not
/// have one: the sandbox it generates, the decisions it makes before Python is
/// involved, the honesty of its unavailability, and the whole of the installer
/// except the two lines that talk to PyPI.
/// </summary>
[Collection(LivePythonCollection.Name)]
public sealed class EmbeddedPythonTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-py-" + Guid.NewGuid().ToString("N"));
    private readonly string _work;
    private readonly string _temp;
    private readonly string _readable;
    private readonly string _packages;
    private readonly string _outside;

    public EmbeddedPythonTests()
    {
        _work = Path.Combine(_root, "work");
        _temp = Path.Combine(_root, "tmp");
        _readable = Path.Combine(_root, "skill");
        _packages = Path.Combine(_root, "packages");
        _outside = Path.Combine(_root, "outside");
        foreach (string directory in new[] { _work, _temp, _readable, _packages, _outside })
            Directory.CreateDirectory(directory);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, true); } catch { }
    }

    private ExecutionPolicy Policy(bool network = false, bool scripts = true, bool packages = true) => new(
        AllowScripts: scripts,
        AllowNetwork: network,
        WorkRoot: _work,
        ReadableRoots: [_readable],
        TempRoot: _temp)
    {
        PackageRoot = packages ? _packages : null,
        DefaultTimeout = TimeSpan.FromSeconds(20),
    };

    private InterpreterContext Context(ExecutionPolicy? policy = null) =>
        new(_work, new Dictionary<string, string> { ["HOME"] = _work }, policy ?? Policy());

    private static string Real(string path) => ConfinedPaths.RealPath(path);

    // =====================================================================================
    // the sandbox source, per policy shape
    // =====================================================================================

    [Fact]
    public void TheSandboxMirrorsTheWritableAndReadableRoots()
    {
        string source = PythonBootstrap.CreateSandboxSource(Policy());

        // Writable is exactly the work and temp roots, as real paths: the hook
        // compares against realpath(), so the literals have to be real too.
        Assert.Contains($"        {PythonBootstrap.Literal(Real(_work))},", source);
        Assert.Contains($"        {PythonBootstrap.Literal(Real(_temp))},", source);
        Assert.Contains($"        {PythonBootstrap.Literal(Real(_readable))},", source);
        Assert.Contains($"        {PythonBootstrap.Literal(Real(_packages))},", source);

        // The read-only roots must not appear in the writable list. Take the
        // slice between the two keywords and look there.
        string writable = Between(source, "writable=[", "]");
        Assert.Contains(PythonBootstrap.Literal(Real(_work)), writable);
        Assert.Contains(PythonBootstrap.Literal(Real(_temp)), writable);
        Assert.DoesNotContain(PythonBootstrap.Literal(Real(_readable)), writable);
        Assert.DoesNotContain(PythonBootstrap.Literal(Real(_packages)), writable);
    }

    [Fact]
    public void TheSandboxCarriesTheNetworkSwitchBothWays()
    {
        Assert.Contains("network=False,", PythonBootstrap.CreateSandboxSource(Policy(network: false)));
        Assert.Contains("network=True,", PythonBootstrap.CreateSandboxSource(Policy(network: true)));
    }

    [Fact]
    public void TheSandboxHasNoPackageRootWhenTheSessionHasNone()
    {
        string source = PythonBootstrap.CreateSandboxSource(Policy(packages: false));
        Assert.DoesNotContain(PythonBootstrap.Literal(Real(_packages)), source);
        Assert.Contains(PythonBootstrap.Literal(Real(_readable)), source);
    }

    [Fact]
    public void PythonPathEntriesAreAppliedInOrderOnlyWhenTheSandboxCanReadThem()
    {
        ExecutionPolicy policy = Policy(packages: false) with
        {
            ReadableRoots = new[] { _readable, _packages },
        };
        string pythonPath = string.Join(
            Path.PathSeparator, _packages, _outside, _readable, _packages);
        var context = new InterpreterContext(
            _work,
            new Dictionary<string, string> { ["PYTHONPATH"] = pythonPath },
            policy);

        IReadOnlyList<string> paths = EmbeddedPython.ReadableImportPaths(
            context, new ConfinedPaths(policy));

        Assert.Equal(new[] { Real(_packages), Real(_readable) }, paths);
    }

    [Fact]
    public void PackageRootRemainsTheImportPathFallbackForDirectRuntimeCalls()
    {
        ExecutionPolicy policy = Policy();
        IReadOnlyList<string> paths = EmbeddedPython.ReadableImportPaths(
            Context(policy), new ConfinedPaths(policy));

        Assert.Equal(new[] { Real(_packages) }, paths);
    }

    [Fact]
    public void TheRunPayloadSeparatesSessionModuleRootsFromTheRuntimePath()
    {
        string payload = PythonInterpreter.CreatePayload(
            "code",
            "pass",
            ["-c"],
            _work,
            new Dictionary<string, string>(),
            [_work, _readable],
            [_packages, _readable],
            Path.Combine(_root, "runtime", "app_packages"),
            TimeSpan.FromSeconds(7),
            null);

        using JsonDocument document = JsonDocument.Parse(payload);
        string[] roots = document.RootElement.GetProperty("module_roots")
            .EnumerateArray()
            .Select(element => element.GetString()!)
            .ToArray();

        Assert.Equal(7, document.RootElement.GetProperty("network_timeout_seconds").GetDouble());
        Assert.Equal(
            Path.Combine(_root, "runtime", "app_packages"),
            document.RootElement.GetProperty("runtime_packages").GetString());

        // cwd/path_front commonly name the same directory, and a skill may occur
        // in both front and back. Each is scanned once, in first-search order.
        Assert.Equal(new[] { _work, _readable, _packages }, roots);
    }

    [Fact]
    public void TheSandboxAddsTheRuntimesOwnTreesToTheReadableRoots()
    {
        string stdlib = Path.Combine(_root, "runtime", "lib", "python3.13");
        Directory.CreateDirectory(stdlib);
        string source = PythonBootstrap.CreateSandboxSource(Policy(), [stdlib]);

        // Without this the interpreter cannot import its own standard library.
        Assert.Contains(PythonBootstrap.Literal(Real(stdlib)), Between(source, "readable=[", "]"));
        Assert.DoesNotContain(PythonBootstrap.Literal(Real(stdlib)), Between(source, "writable=[", "]"));
    }

    [Fact]
    public void TheSandboxIsInstalledAsAnAuditHookAndResolvesSymlinks()
    {
        string source = PythonBootstrap.CreateSandboxSource(Policy());
        Assert.Contains("sys.addaudithook(_hook)", source);
        // A prefix test on the literal path would be defeated by a symlink; the
        // hook has to resolve first, exactly as ConfinedPaths does.
        Assert.Contains("os.path.realpath(text)", source);
        // Fails closed: a run that never got a policy can touch nothing.
        Assert.Contains("'writable': (), 'readable': (), 'network': False", source);
    }

    [Fact]
    public void TheSandboxRefusesChildProcessesUnderEveryPolicy()
    {
        foreach (ExecutionPolicy policy in new[] { Policy(network: false), Policy(network: true) })
        {
            string source = PythonBootstrap.CreateSandboxSource(policy);
            foreach (string denied in new[] { "'os.system'", "'os.exec'", "'os.fork'", "'os.posix_spawn'", "'os.spawn'", "'subprocess.Popen'" })
                Assert.Contains(denied, source);
            Assert.Contains(ExecutionPolicy.ProcessesUnavailableMessage, source);
            Assert.Contains("iOS does not let an app start another program", source);
        }
    }

    [Fact]
    public void TheSandboxRefusesNativeLibraryLoadsUnderEveryPolicy()
    {
        foreach (ExecutionPolicy policy in new[] { Policy(network: false), Policy(network: true) })
        {
            string source = PythonBootstrap.CreateSandboxSource(policy);
            // Symbol lookup and the call gate stay refused outright: a handle that
            // somehow appears still cannot be used to reach native code.
            Assert.Contains("'ctypes.dlsym'", source);
            Assert.Contains("'ctypes.call_function'", source);
            Assert.Contains("step around every other check", source);
        }
    }

    [Fact]
    public void ALibraryLoadIsAllowedOnlyFromTheAppsOwnBundle()
    {
        // A blanket refusal of dlopen does not stop an attacker on this platform: it
        // stops `import numpy`. Every compiled extension module is a signed framework
        // that the import machinery loads by dlopen, so the rule has to be about which
        // library, and the bundle root is what decides.
        string? previous = PythonBootstrap.BundleRoot;
        try
        {
            PythonBootstrap.BundleRoot = Path.Combine(_root, "bundle");
            Directory.CreateDirectory(PythonBootstrap.BundleRoot);
            string source = PythonBootstrap.CreateSandboxSource(Policy());

            Assert.Contains("def _check_dlopen", source);
            Assert.Contains("bundle=", source);
            Assert.Contains(PythonBootstrap.Literal(PythonBootstrap.BundleRoot), source);
            // The decision is made on the resolved path, so a symlink laid inside the
            // bundle cannot point at a library outside it.
            Assert.Contains("os.path.realpath", source);
            Assert.Contains("text.startswith(bundle + os.sep)", source);
        }
        finally
        {
            PythonBootstrap.BundleRoot = previous;
        }
    }

    [Fact]
    public void AHandleToTheMainProgramIsAllowedBecauseItLoadsNothing()
    {
        // dlopen(NULL) returns a handle to the image already running. numpy and
        // Pillow both ask for one while probing during import, and refusing it makes
        // them unimportable while preventing nothing: no new code enters the process.
        string source = PythonBootstrap.CreateSandboxSource(Policy());
        Assert.Contains("if name is None or name == '':", source);
        // What keeps that safe is the pair that is still refused outright.
        Assert.Contains("'ctypes.dlsym'", source);
        Assert.Contains("'ctypes.call_function'", source);
    }

    [Fact]
    public void WithNoBundleEveryLibraryLoadStaysRefused()
    {
        string? previous = PythonBootstrap.BundleRoot;
        try
        {
            PythonBootstrap.BundleRoot = null;
            string source = PythonBootstrap.CreateSandboxSource(Policy());
            Assert.Contains("bundle=''", source);
            // With no bundle the prefix test can never pass, so every named library is
            // refused and the message says why rather than leaving the reader guessing.
            Assert.Contains("if bundle and (text == bundle or text.startswith(bundle + os.sep)):", source);
            Assert.Contains("This host named no bundle", source);
        }
        finally
        {
            PythonBootstrap.BundleRoot = previous;
        }
    }

    [Fact]
    public void TheSandboxUsesTheSharedWordingWhenTheNetworkIsOff()
    {
        string source = PythonBootstrap.CreateSandboxSource(Policy(network: false));
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, source);
        // The families that reach a socket one way or another.
        foreach (string family in new[] { "'socket.'", "'urllib.'", "'http.client.'" })
            Assert.Contains(family, source);
    }

    [Fact]
    public void TheSandboxChecksTheCallsThatOpenNothing()
    {
        // open() is not the only way to touch a file; these are audited too and
        // a sandbox that only watched open() would be trivially walked around.
        string source = PythonBootstrap.CreateSandboxSource(Policy());
        foreach (string audited in new[] { "'os.mkdir'", "'os.remove'", "'os.rename'", "'os.symlink'", "'os.listdir'", "'os.chdir'" })
            Assert.Contains(audited, source);
    }

    [Fact]
    public void PathLiteralsSurviveQuotesAndBackslashes()
    {
        Assert.Equal(@"'a\\b'", PythonBootstrap.Literal(@"a\b"));
        Assert.Equal(@"'it\'s'", PythonBootstrap.Literal("it's"));
        Assert.Equal(@"'a\nb'", PythonBootstrap.Literal("a\nb"));
    }

    [Fact]
    public void TheBootstrapForgetsOnlyModulesOwnedByTheCompletedRun()
    {
        string source = PythonBootstrap.InstallSource;
        string lifetime = Between(
            source,
            "# ---------------------------------------------------- module lifetime",
            "# ------------------------------------------------------------- run");

        Assert.Contains("request.get('module_roots', ())", lifetime);
        Assert.Contains("_state['writable']", lifetime);
        // readable also contains the runtime's stdlib and bundled packages. Using
        // it here would turn every command into a cold interpreter in disguise.
        Assert.DoesNotContain("_state['readable']", lifetime);
        Assert.Contains("namespace.get('__file__')", lifetime);
        Assert.Contains("getattr(spec, 'origin', None)", lifetime);
        Assert.Contains("getattr(module, '__path__', None)", lifetime);
        Assert.Contains("sys.modules.pop(name, None)", lifetime);
        Assert.Contains("sys.path_importer_cache.pop(entry, None)", lifetime);
        Assert.Contains("invalidate = getattr(finder, 'invalidate_caches', None)", lifetime);
        Assert.DoesNotContain("importlib.invalidate_caches()", lifetime);
        Assert.DoesNotContain("_guard.busy = True", lifetime);
        Assert.Contains("previous_modules.get(name) is module", lifetime);
        Assert.Contains("_trusted_modules", lifetime);
        Assert.Contains("_forget_shadowed_runtime_modules", lifetime);
        Assert.Contains("_module_belongs_to_run(_module_signature(module), (bundled,))", lifetime);
        Assert.Contains("_forget_run_modules(module_roots, previous_modules)", source);
        Assert.Contains("_forget_shadowed_runtime_modules(front + back, runtime_packages)", source);
    }

    // =====================================================================================
    // the path decision the host makes before Python is involved
    // =====================================================================================

    [Fact]
    public void APathUnderTheWorkRootIsReadableAndWritable()
    {
        var confined = new ConfinedPaths(Policy());
        File.WriteAllText(Path.Combine(_work, "a.txt"), "x");
        Assert.True(PythonBootstrap.IsPathAllowed(confined, "a.txt", _work, forWrite: false));
        Assert.True(PythonBootstrap.IsPathAllowed(confined, "a.txt", _work, forWrite: true));
        Assert.True(PythonBootstrap.IsPathAllowed(confined, "new.txt", _work, forWrite: true));
    }

    [Fact]
    public void AReadOnlyRootIsReadableButNotWritable()
    {
        var confined = new ConfinedPaths(Policy());
        string file = Path.Combine(_readable, "skill.py");
        File.WriteAllText(file, "x");
        Assert.True(PythonBootstrap.IsPathAllowed(confined, file, _work, forWrite: false));
        Assert.False(PythonBootstrap.IsPathAllowed(confined, file, _work, forWrite: true));
    }

    [Fact]
    public void APathOutsideEveryRootIsRefused()
    {
        var confined = new ConfinedPaths(Policy());
        string file = Path.Combine(_outside, "secret.txt");
        File.WriteAllText(file, "x");
        Assert.False(PythonBootstrap.IsPathAllowed(confined, file, _work, forWrite: false));
        Assert.False(PythonBootstrap.IsPathAllowed(confined, "/etc/hosts", _work, forWrite: false));
    }

    [SkippableFact]
    public void ASymlinkOutOfTheSandboxIsRefused()
    {
        // The reason the check resolves before it compares: this path is
        // lexically inside the work root and physically is not.
        string target = Path.Combine(_outside, "secret.txt");
        File.WriteAllText(target, "x");
        TestPlatforms.CreateFileSymlink(Path.Combine(_work, "link.txt"), target);

        var confined = new ConfinedPaths(Policy());
        Assert.False(PythonBootstrap.IsPathAllowed(confined, "link.txt", _work, forWrite: false));
        Assert.False(PythonBootstrap.IsPathAllowed(confined, "link.txt", _work, forWrite: true));
    }

    [SkippableFact]
    public void ADirectorySymlinkOutOfTheSandboxIsRefused()
    {
        TestPlatforms.CreateDirectorySymlink(Path.Combine(_work, "escape"), _outside);
        var confined = new ConfinedPaths(Policy());
        Assert.False(PythonBootstrap.IsPathAllowed(confined, "escape/new.txt", _work, forWrite: true));
    }

    [Fact]
    public void TheRefusalNamesTheRootsTheRunActuallyHas()
    {
        var confined = new ConfinedPaths(Policy());
        string message = PythonBootstrap.DeniedMessage("/etc/hosts", forWrite: false, confined.ReadableRoots);
        Assert.Contains("/etc/hosts: Permission denied", message);
        Assert.Contains(Real(_work), message);
        Assert.Contains("read under", message);
        Assert.Contains("write under", PythonBootstrap.DeniedMessage("/etc/x", true, confined.WritableRoots));
        Assert.Contains("may not touch the file system", PythonBootstrap.DeniedMessage("/etc/x", true, []));
    }

    // =====================================================================================
    // honesty about not being there
    // =====================================================================================

    [Fact]
    public void AMissingRuntimeRootIsReportedByName()
    {
        string missing = Path.Combine(_root, "no-such-runtime");
        var python = new EmbeddedPython(missing);

        Assert.False(python.IsAvailable);
        Assert.NotNull(python.UnavailableReason);
        Assert.Contains(missing, python.UnavailableReason);
        Assert.Contains("does not exist", python.UnavailableReason);
        Assert.Equal(string.Empty, python.Version);
    }

    [Fact]
    public void ARootWithoutAStandardLibraryNamesWhatItLooksFor()
    {
        // The directory is there; what makes a Python runtime is not.
        string empty = Path.Combine(_root, "empty-runtime");
        Directory.CreateDirectory(empty);
        var python = new EmbeddedPython(empty);

        Assert.False(python.IsAvailable);
        Assert.NotNull(python.UnavailableReason);
        Assert.Contains(empty, python.UnavailableReason);
        Assert.Contains("tensoragent-python.json", python.UnavailableReason);
        Assert.Contains("lib/python3.13", python.UnavailableReason);
    }

    [Fact]
    public void AManifestPointingAtNothingIsReportedRatherThanGuessedAround()
    {
        string root = Path.Combine(_root, "broken-runtime");
        Directory.CreateDirectory(Path.Combine(root, "python"));
        File.WriteAllText(
            Path.Combine(root, "python", "tensoragent-python.json"),
            """{ "version": "3.13", "stdlib": "python/lib/python3.13", "packages": "python/app_packages", "layout": 1 }""");

        var python = new EmbeddedPython(root);
        Assert.False(python.IsAvailable);
        Assert.Contains("is not there", python.UnavailableReason!);
        Assert.Contains("python3.13", python.UnavailableReason!);
    }

    [Fact]
    public async Task RunningWithoutAnInterpreterFailsWithTheReasonAndCommandNotFound()
    {
        var python = new EmbeddedPython(Path.Combine(_root, "no-such-runtime"));
        ExecutionResult result = await python.RunCodeAsync("print(1)", [], Context(), CancellationToken.None);

        Assert.Equal(ExecutionResult.CommandNotFoundExitCode, result.ExitCode);
        Assert.False(result.TimedOut);
        Assert.Contains("does not exist", result.Stderr);
        Assert.Equal(string.Empty, result.Stdout);
    }

    [Fact]
    public async Task RunningIsRefusedBeforeAnythingElseWhenScriptsAreOff()
    {
        // Checked ahead of availability on purpose: "scripts are off" is the
        // true answer even on a host that does have an interpreter.
        var python = new EmbeddedPython(Path.Combine(_root, "no-such-runtime"));
        ExecutionResult result = await python.RunCodeAsync("print(1)", [], Context(Policy(scripts: false)), CancellationToken.None);

        Assert.Equal(126, result.ExitCode);
        Assert.Contains(ExecutionPolicy.ScriptsDisabledMessage, result.Stderr);
    }

    [Fact]
    public async Task ASyntaxCheckWithoutAnInterpreterSaysSoRatherThanPassing()
    {
        var python = new EmbeddedPython(Path.Combine(_root, "no-such-runtime"));
        string script = Path.Combine(_work, "s.py");
        File.WriteAllText(script, "print(");

        SyntaxCheckResult result = await python.CheckSyntaxAsync(script, CancellationToken.None);
        Assert.False(result.Ok);
        Assert.Contains(script, result.Message!);
        Assert.Contains("does not exist", result.Message!);
    }

    /// <summary>
    /// Every caller of an interpreter that is still starting must be made to wait for
    /// the answer, not handed "there is none".
    ///
    /// <para>
    /// This is the bug behind "the model refuses to use the shell tool". The
    /// initialization flag was published BEFORE the work rather than after it, so the
    /// fast path outside the lock was a window into a half-started interpreter: a second
    /// caller arriving while <c>Py_Initialize</c> was running saw "already tried", found
    /// no interpreter and no reason, and reported unavailable. The app opens that window
    /// on every launch — the page fetches <c>/api/agent/engine</c> as it loads, which
    /// asks for the version and starts CPython, while the startup self-test runs
    /// <c>python3</c> on its own thread — and it was observed doing exactly that in the
    /// simulator: four self-test checks saying "no Python interpreter is embedded in this
    /// build" from a build whose engine line, moments later, read "python 3.13.14". A
    /// model told that on the first command of a session stops reaching for the shell.
    /// </para>
    /// <para>
    /// The invariant asserted here is the one that was broken and holds whether or not a
    /// real runtime is present: unavailable is never silent. Every observation either
    /// finds an interpreter or says why there is none.
    /// </para>
    /// </summary>
    [Fact]
    public void AskingFromEveryThreadAtOnceNeverProducesASilentUnavailable()
    {
        var python = new EmbeddedPython(Path.Combine(_root, "no-such-runtime"));
        var answers = new System.Collections.Concurrent.ConcurrentBag<(bool Available, string? Reason)>();

        using var start = new Barrier(32);
        Parallel.For(0, 32, _ =>
        {
            start.SignalAndWait();
            answers.Add((python.IsAvailable, python.UnavailableReason));
        });

        Assert.Equal(32, answers.Count);
        foreach ((bool available, string? reason) in answers)
        {
            Assert.False(available);
            Assert.False(string.IsNullOrWhiteSpace(reason),
                "a caller was told there is no interpreter and given no reason, which is the "
                + "shape of a half-initialized runtime being reported as a missing one");
        }
    }

    /// <summary>
    /// The same race with a real interpreter, where the window is hundreds of
    /// milliseconds wide instead of microseconds: every thread must see the one that
    /// started.
    /// </summary>
    [LivePythonFact]
    public void AllOfThemSeeTheInterpreterThatOneOfThemStarted()
    {
        var python = new EmbeddedPython(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable));
        var seen = new System.Collections.Concurrent.ConcurrentBag<bool>();

        using var start = new Barrier(16);
        Parallel.For(0, 16, _ =>
        {
            start.SignalAndWait();
            seen.Add(python.IsAvailable);
        });

        Assert.Equal(16, seen.Count);
        Assert.DoesNotContain(false, seen);
        Assert.True(python.IsAvailable, python.UnavailableReason);
    }

    // =====================================================================================
    // the installer
    // =====================================================================================

    [Fact]
    public void TheInstallerCannotInstallWithoutNetworkPermission()
    {
        var installer = new WheelInstaller(Policy(network: false));
        Assert.False(installer.CanInstall);
        Assert.Equal(ExecutionPolicy.NetworkDisabledMessage, installer.UnavailableReason);
    }

    [Fact]
    public void TheInstallerCannotInstallWhenTheIndexIsNotAnAllowedHost()
    {
        var installer = new WheelInstaller(Policy(network: true) with { NetworkHosts = ["example.com"] });
        Assert.False(installer.CanInstall);
        Assert.Contains("pypi.org", installer.UnavailableReason!);
    }

    [Fact]
    public void TheInstallerDoesNotAdvertiseInstallsWhenPyPisPayloadHostIsBlocked()
    {
        var installer = new WheelInstaller(Policy(network: true) with { NetworkHosts = [WheelInstaller.IndexHost] });
        Assert.False(installer.CanInstall);
        Assert.Contains(WheelInstaller.PayloadHost, installer.UnavailableReason!);
    }

    [Fact]
    public async Task ADirectInstallDoesNotFetchMetadataWhenPyPisPayloadHostIsBlocked()
    {
        var handler = new RecordingHttpHandler(_ => TextResponse(HttpStatusCode.OK, "unexpected request"));
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = [WheelInstaller.IndexHost] };
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains(WheelInstaller.PayloadHost, result.Stderr);
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public void TheInstallerCanInstallWhenTheNetworkIsOn()
    {
        var installer = new WheelInstaller(Policy(network: true));
        Assert.True(installer.CanInstall);
        Assert.Null(installer.UnavailableReason);
    }

    [Fact]
    public async Task TheInstallerRefusesToInstallWithoutNetworkPermission()
    {
        var installer = new WheelInstaller(Policy(network: false));
        var request = new InstallRequest("python", ["rich"], _packages, Policy(network: false));

        ExecutionResult result = await installer.InstallAsync(request, CancellationToken.None);
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task TheInstallerRefusesALanguageItDoesNotInstall()
    {
        var installer = new WheelInstaller(Policy(network: true));
        var request = new InstallRequest("node", ["left-pad"], _packages, Policy(network: true));

        ExecutionResult result = await installer.InstallAsync(request, CancellationToken.None);
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("only handles Python packages", result.Stderr);
    }

    [Fact]
    public async Task TheInstallerRefusesAHostThePolicyDoesNotAllow()
    {
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = ["example.com"] };
        var installer = new WheelInstaller(policy);
        var request = new InstallRequest("python", ["rich"], _packages, policy);

        ExecutionResult result = await installer.InstallAsync(request, CancellationToken.None);
        Assert.NotEqual(0, result.ExitCode);
        Assert.Contains("pypi.org", result.Stderr);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TheInstallerNeverRequestsADisallowedWheelOrRedirectHost(bool throughRedirect)
    {
        const string index = "https://pypi.org/pypi/probe/json";
        const string allowedPayload = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        const string refusedPayload = "https://downloads.example/probe-1.0-py3-none-any.whl";
        byte[] wheel = WheelBytes(("probe.py", "answer = 42\n"));
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();

        var handler = new RecordingHttpHandler(request =>
        {
            string uri = request.RequestUri!.AbsoluteUri;
            if (uri == index)
            {
                string payload = throughRedirect ? allowedPayload : refusedPayload;
                return TextResponse(HttpStatusCode.OK, IndexJson(
                    IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", digest, url: payload)));
            }
            if (throughRedirect && uri == allowedPayload)
            {
                var redirect = new HttpResponseMessage(HttpStatusCode.Found);
                redirect.Headers.Location = new Uri(refusedPayload);
                return redirect;
            }
            return TextResponse(HttpStatusCode.InternalServerError, "unexpected request");
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true) with
        {
            NetworkHosts = ["pypi.org", "files.pythonhosted.org"],
        };
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains("downloads.example", result.Stderr, StringComparison.Ordinal);
        Assert.DoesNotContain(handler.Requests, uri => uri.Host == "downloads.example");
        Assert.Equal(throughRedirect ? 2 : 1, handler.Requests.Count);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task TheInstallerFollowsRedirectsOnlyAfterEachHostIsAllowed()
    {
        const string index = "https://pypi.org/pypi/probe/json";
        const string origin = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        const string mirror = "https://wheel-cdn.example/probe-1.0-py3-none-any.whl";
        byte[] wheel = WheelBytes(
            ("probe.py", "answer = 42\n"),
            ("probe-1.0.dist-info/METADATA", "Name: probe\nVersion: 1.0\n"));
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();

        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", digest, url: origin))),
            origin => RedirectResponse(mirror),
            mirror => BytesResponse(wheel),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true) with
        {
            NetworkHosts = ["pypi.org", "files.pythonhosted.org", "wheel-cdn.example"],
        };
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.True(result.Ok, result.Stderr);
        Assert.Equal(new[] { new Uri(index), new Uri(origin), new Uri(mirror) }, handler.Requests);
        Assert.True(File.Exists(Path.Combine(_packages, "probe.py")));
    }

    [Fact]
    public async Task PackageMetadataHasABoundedStreamingSize()
    {
        const string index = "https://pypi.org/pypi/probe/json";
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri == index
            ? DeclaredLengthResponse(WheelInstaller.MaxIndexBytes + 1L)
            : TextResponse(HttpStatusCode.InternalServerError, "unexpected request"));
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains("package metadata exceeds the 8 MiB limit", result.Stderr);
        Assert.Single(handler.Requests);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task ASuccessfulInstallTruthfullyReportsUnresolvedUnconditionalDependencies()
    {
        byte[] wheel = WheelBytes(("probe.py", "VALUE = 1\n"));
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();
        const string index = "https://pypi.org/pypi/probe/json";
        const string payload = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJsonWithDependencies(
                [
                    "Pillow",
                    "pikepdf>=8",
                    "typing-extensions; python_version < '3.11'",
                    "platformdirs; python_version >= '3.13'",
                ],
                IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", digest, url: payload))),
            payload => BytesResponse(wheel),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.True(result.Ok, result.Stderr);
        Assert.Contains("declares these unconditional dependencies (Pillow, pikepdf)", result.Stdout);
        Assert.Contains("did not resolve them automatically", result.Stdout);
        Assert.Contains("Conditional dependency markers were not evaluated or shown", result.Stdout);
        Assert.DoesNotContain("typing-extensions", result.Stdout);
        Assert.DoesNotContain("platformdirs", result.Stdout);
        Assert.True(File.Exists(Path.Combine(_packages, "probe.py")));
    }

    [Fact]
    public async Task AWheelDownloadHasABoundedStreamingSizeAndLeavesNoStage()
    {
        const string index = "https://pypi.org/pypi/probe/json";
        const string payload = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", "abcd", url: payload))),
            payload => DeclaredLengthResponse(WheelInstaller.MaxWheelBytes + 1),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains("wheel download exceeds the 64 MiB limit", result.Stderr);
        Assert.Equal(2, handler.Requests.Count);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task CancellingAStreamingDownloadPreservesTheExistingEnvironmentAndRemovesTheStage()
    {
        string existing = Path.Combine(_packages, "existing.py");
        File.WriteAllText(existing, "VALUE = 'safe'\n");
        byte[] wheel = WheelBytes(("probe.py", "VALUE = 'partial'\n"));
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();
        const string index = "https://pypi.org/pypi/probe/json";
        const string payload = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        using var cancellation = new CancellationTokenSource();
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", digest, url: payload))),
            payload => StreamResponse(new CancelOnFirstReadStream(wheel, cancellation)),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), cancellation.Token));

        Assert.Equal("VALUE = 'safe'\n", File.ReadAllText(existing));
        Assert.Equal([existing], Directory.GetFiles(_packages));
        Assert.Empty(Directory.GetDirectories(_packages));
    }

    [Fact]
    public async Task ReplacingAPinnedWheelRemovesItsOldRecordFilesAndMetadataButPreservesNeighbors()
    {
        string module = Path.Combine(_packages, "probe");
        string oldInfo = Path.Combine(_packages, "probe-1.0.dist-info");
        Directory.CreateDirectory(module);
        Directory.CreateDirectory(oldInfo);
        File.WriteAllText(Path.Combine(module, "__init__.py"), "VERSION = '1.0'\n");
        File.WriteAllText(Path.Combine(module, "obsolete.py"), "OLD = True\n");
        File.WriteAllText(Path.Combine(module, "neighbor.py"), "KEEP = True\n");
        File.WriteAllText(Path.Combine(oldInfo, "METADATA"), "Name: probe\nVersion: 1.0\n");
        File.WriteAllText(Path.Combine(oldInfo, "RECORD"),
            "probe/__init__.py,,\nprobe/obsolete.py,,\n"
            + "probe-1.0.dist-info/METADATA,,\nprobe-1.0.dist-info/RECORD,,\n");

        byte[] wheel = WheelBytes(
            ("probe/__init__.py", "VERSION = '2.0'\n"),
            ("probe/current.py", "CURRENT = True\n"),
            ("probe-2.0.dist-info/METADATA", "Name: probe\nVersion: 2.0\n"),
            ("probe-2.0.dist-info/RECORD", "probe/__init__.py,,\nprobe/current.py,,\n"));
        const string index = "https://pypi.org/pypi/probe/2.0/json";
        const string payload = "https://files.pythonhosted.org/packages/probe-2.0-py3-none-any.whl";
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("probe-2.0-py3-none-any.whl", "bdist_wheel", digest, url: payload))),
            payload => BytesResponse(wheel),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe==2.0"], _packages, policy), CancellationToken.None);

        Assert.True(result.Ok, result.Stderr);
        Assert.Equal("VERSION = '2.0'\n", File.ReadAllText(Path.Combine(module, "__init__.py")));
        Assert.True(File.Exists(Path.Combine(module, "current.py")));
        Assert.True(File.Exists(Path.Combine(module, "neighbor.py")));
        Assert.False(File.Exists(Path.Combine(module, "obsolete.py")));
        Assert.False(Directory.Exists(oldInfo));
        Assert.True(Directory.Exists(Path.Combine(_packages, "probe-2.0.dist-info")));
        Assert.DoesNotContain(Directory.GetFileSystemEntries(_packages),
            path => Path.GetFileName(path).StartsWith(".tensoragent-install-", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFailedLaterPackageDoesNotRollBackAnEarlierPackageOrLeaveItsOwnFiles()
    {
        byte[] alpha = WheelBytes(("alpha.py", "VALUE = 1\n"));
        byte[] bravo = WheelBytes(("bravo.py", "VALUE = 2\n"));
        string alphaDigest = Convert.ToHexString(SHA256.HashData(alpha)).ToLowerInvariant();
        const string alphaIndex = "https://pypi.org/pypi/alpha/json";
        const string bravoIndex = "https://pypi.org/pypi/bravo/json";
        const string alphaPayload = "https://files.pythonhosted.org/packages/alpha-1.0-py3-none-any.whl";
        const string bravoPayload = "https://files.pythonhosted.org/packages/bravo-1.0-py3-none-any.whl";
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            alphaIndex => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("alpha-1.0-py3-none-any.whl", "bdist_wheel", alphaDigest, url: alphaPayload))),
            bravoIndex => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("bravo-1.0-py3-none-any.whl", "bdist_wheel", new string('0', 64), url: bravoPayload))),
            alphaPayload => BytesResponse(alpha),
            bravoPayload => BytesResponse(bravo),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client);

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["alpha", "bravo"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains("installed alpha-1.0-py3-none-any.whl", result.Stdout);
        Assert.Contains("does not match the sha256", result.Stderr);
        Assert.True(File.Exists(Path.Combine(_packages, "alpha.py")));
        Assert.False(File.Exists(Path.Combine(_packages, "bravo.py")));
        Assert.DoesNotContain(Directory.GetFileSystemEntries(_packages),
            path => Path.GetFileName(path).StartsWith(".tensoragent-install-", StringComparison.Ordinal));
    }

    [Fact]
    public void TheInstallerPicksThePurePythonWheelAndCarriesItsDigest()
    {
        Assert.True(WheelInstaller.TrySelectWheel(IndexJson(
            IndexFile("rich-13.9.4.tar.gz", "sdist", "aaa"),
            IndexFile("rich-13.9.4-py3-none-any.whl", "bdist_wheel", "bbb")), out PyPiFile? wheel, out string? reason));

        Assert.Null(reason);
        Assert.NotNull(wheel);
        Assert.Equal("rich-13.9.4-py3-none-any.whl", wheel.FileName);
        Assert.Equal("bbb", wheel.Sha256);
    }

    [Fact]
    public void APackageThatShipsOnlyCompiledCodeIsReportedBeforeAnythingIsInstalled()
    {
        // What a hub skill's compatibility check asks before it is installed on the device:
        // the same index entry and the same selection an install makes, nothing downloaded.
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsolutePath switch
        {
            "/pypi/pandas/json" => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("pandas-2.3.0.tar.gz", "sdist", "aaa"),
                IndexFile("pandas-2.3.0-cp313-cp313-macosx_11_0_arm64.whl", "bdist_wheel", "bbb"))),
            "/pypi/yfinance/json" => TextResponse(HttpStatusCode.OK, IndexJson(
                IndexFile("yfinance-1.8.0-py3-none-any.whl", "bdist_wheel", "ccc"))),
            _ => TextResponse(HttpStatusCode.NotFound, "{}"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true) with { NetworkHosts = ["pypi.org", "files.pythonhosted.org"] };
        var installer = new InstallHookPackageInstaller(new WheelInstaller(policy, client),
            new CodeExecOptions { Enabled = true, AllowInstall = true, AllowNetwork = true }, () => policy.NetworkHosts,
            () => [new BundledDistribution("numpy", "numpy", "2.5.2", Compiled: true)]);

        string? pandas = installer.ExplainUnavailable(CodeLanguage.Python, "pandas");
        Assert.NotNull(pandas);
        Assert.StartsWith("pandas: ", pandas, StringComparison.Ordinal);
        Assert.Contains("compiled code", pandas, StringComparison.Ordinal);
        Assert.Null(installer.ExplainUnavailable(CodeLanguage.Python, "yfinance"));
        // Built into the app: usable, and no lookup.
        Assert.Null(installer.ExplainUnavailable(CodeLanguage.Python, "numpy"));
        // Unknown is not "no": a lookup that fails refuses nothing.
        Assert.Null(installer.ExplainUnavailable(CodeLanguage.Python, "does-not-exist"));
        Assert.DoesNotContain(handler.Requests, r => r.AbsolutePath.Contains("numpy", StringComparison.Ordinal));
        Assert.DoesNotContain(handler.Requests, r => r.Host == WheelInstaller.PayloadHost);
    }

    [Fact]
    public void TheInstallerRefusesAWheelThatWouldNeedACompilerAndSaysWhy()
    {
        Assert.False(WheelInstaller.TrySelectWheel(IndexJson(
            IndexFile("numpy-2.5.2.tar.gz", "sdist", "aaa"),
            IndexFile("numpy-2.5.2-cp313-cp313-macosx_11_0_arm64.whl", "bdist_wheel", "bbb"),
            IndexFile("numpy-2.5.2-cp313-cp313-manylinux_2_17_x86_64.whl", "bdist_wheel", "ccc")), out PyPiFile? wheel, out string? reason));

        Assert.Null(wheel);
        Assert.Contains("compiled code", reason!);
        Assert.Contains("py3-none-any", reason!);
        Assert.Contains("signed into the app at build time", reason!);
    }

    [Fact]
    public void TheInstallerRefusesASourceOnlyPackageAndSaysACompilerIsMissing()
    {
        Assert.False(WheelInstaller.TrySelectWheel(
            IndexJson(IndexFile("pyyaml-6.0.tar.gz", "sdist", "aaa")), out PyPiFile? wheel, out string? reason));

        Assert.Null(wheel);
        Assert.Contains("only a source distribution", reason!);
        Assert.Contains("compiler", reason!);
    }

    [Fact]
    public void TheInstallerSkipsAYankedWheel()
    {
        Assert.True(WheelInstaller.TrySelectWheel(IndexJson(
            IndexFile("rich-1.0-py3-none-any.whl", "bdist_wheel", "aaa", yanked: true),
            IndexFile("rich-1.1-py3-none-any.whl", "bdist_wheel", "bbb")), out PyPiFile? wheel, out _));
        Assert.Equal("rich-1.1-py3-none-any.whl", wheel!.FileName);
    }

    [Fact]
    public void TheInstallerRefusesAWheelPublishedWithoutADigest()
    {
        Assert.False(WheelInstaller.TrySelectWheel(
            IndexJson(IndexFile("rich-1.0-py3-none-any.whl", "bdist_wheel", "")), out _, out string? reason));
        Assert.Contains("without a sha256", reason!);
    }

    [Theory]
    [InlineData("rich-13.9.4-py3-none-any.whl", true)]
    [InlineData("six-1.16.0-py2.py3-none-any.whl", true)]
    [InlineData("probe-1.0-cp313-none-any.whl", true)]
    [InlineData("probe-1.0-py36-none-any.whl", true)]
    [InlineData("probe-1.0-py310-none-any.whl", true)]
    [InlineData("probe-1.0-py313-none-any.whl", true)]
    [InlineData("probe-1.0-py314-none-any.whl", false)]
    [InlineData("probe-1.0-cp310-none-any.whl", false)]
    [InlineData("numpy-2.5.2-cp313-cp313-macosx_11_0_arm64.whl", false)]
    [InlineData("pillow-10.4.0-cp313-cp313-ios_13_0_arm64_iphoneos.whl", false)]
    [InlineData("pyyaml-6.0.tar.gz", false)]
    public void OnlyPureWheelsCompatibleWithEmbeddedPythonAreAccepted(string fileName, bool accepted)
        => Assert.Equal(accepted, WheelInstaller.IsPureWheel(fileName));

    [Fact]
    public void APureWheelForAnotherPythonVersionGetsAnInterpreterSpecificRefusal()
    {
        Assert.False(WheelInstaller.TrySelectWheel(IndexJson(IndexFile(
            "probe-1.0-py314-none-any.whl", "bdist_wheel", "aaa")), out _, out string? reason));

        Assert.Contains("pure-Python wheel targets another Python version", reason!);
        Assert.Contains("CPython 3.13", reason!);
    }

    [Theory]
    [InlineData(">=3.5", true)]
    [InlineData(">=3.10,<4", true)]
    [InlineData(">=3.14", false)]
    [InlineData("~=3.8", true)]
    [InlineData("~=3.13.1", true)]
    [InlineData("==3.13.*", true)]
    [InlineData("!=3.13.*", false)]
    [InlineData("==3.13.14", true)]
    [InlineData("!=3.13.0", true)]
    public void RequiresPythonIsEvaluatedForEmbeddedCpython(string expression, bool compatible)
    {
        Assert.True(WheelInstaller.TryEvaluateRequiresPython(expression, out bool actual, out string? error), error);
        Assert.Equal(compatible, actual);
    }

    [Fact]
    public void AProjectIncompatibleWithEmbeddedPythonIsRefusedClearly()
    {
        Assert.False(WheelInstaller.TrySelectWheel(
            IndexJsonWithRequiresPython(">=3.14", IndexFile(
                "probe-1.0-py3-none-any.whl", "bdist_wheel", "aaa")),
            out _, out string? reason));

        Assert.Contains("requires Python '>=3.14'", reason!);
        Assert.Contains("CPython 3.13", reason!);
    }

    [Fact]
    public async Task InstallerUsesTheSuppliedInterpreterPatchForRequiresPython()
    {
        byte[] wheel = WheelBytes(("probe.py", "VALUE = 1\n"));
        string digest = Convert.ToHexString(SHA256.HashData(wheel)).ToLowerInvariant();
        const string index = "https://pypi.org/pypi/probe/json";
        const string payload = "https://files.pythonhosted.org/packages/probe-1.0-py3-none-any.whl";
        var handler = new RecordingHttpHandler(request => request.RequestUri!.AbsoluteUri switch
        {
            index => TextResponse(HttpStatusCode.OK, IndexJsonWithRequiresPython(
                ">=3.13.10", IndexFile(
                    "probe-1.0-py3-none-any.whl", "bdist_wheel", digest, url: payload))),
            payload => BytesResponse(wheel),
            _ => TextResponse(HttpStatusCode.InternalServerError, "unexpected request"),
        });
        using var client = new HttpClient(handler);
        ExecutionPolicy policy = Policy(network: true);
        var installer = new WheelInstaller(policy, client, pythonVersion: "3.13.9 (embedded)");

        ExecutionResult result = await installer.InstallAsync(
            new InstallRequest("python", ["probe"], _packages, policy), CancellationToken.None);

        Assert.False(result.Ok);
        Assert.Contains("requires Python '>=3.13.10'", result.Stderr);
        Assert.Contains("CPython 3.13.9", result.Stderr);
        Assert.Equal(new[] { new Uri(index) }, handler.Requests);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public void OversizedRequiresPythonMetadataCannotFloodOrInjectIntoTheDiagnostic()
    {
        string untrusted = new string('9', 512) + "\nforged second line";

        Assert.False(WheelInstaller.TrySelectWheel(
            IndexJsonWithRequiresPython(untrusted, IndexFile(
                "probe-1.0-py3-none-any.whl", "bdist_wheel", "aaa")),
            out _, out string? reason));

        Assert.NotNull(reason);
        Assert.True(reason!.Length < 700, $"remote metadata expanded the diagnostic to {reason.Length} characters");
        Assert.DoesNotContain('\n', reason);
        Assert.DoesNotContain("forged second line", reason, StringComparison.Ordinal);
        Assert.Contains('…', reason);
    }

    [Fact]
    public void AnIncompatibleFileIsSkippedForACompatiblePureWheel()
    {
        Assert.True(WheelInstaller.TrySelectWheel(IndexJson(
            IndexFile("probe-1.0-py3-none-any.whl", "bdist_wheel", "aaa", requiresPython: ">=3.14"),
            IndexFile("probe-1.0-cp313-none-any.whl", "bdist_wheel", "bbb", requiresPython: ">=3.10")),
            out PyPiFile? wheel, out string? reason), reason);

        Assert.Equal("probe-1.0-cp313-none-any.whl", wheel!.FileName);
    }

    [Fact]
    public void UnsupportedRequiresPythonSyntaxIsRefusedInsteadOfGuessed()
    {
        Assert.False(WheelInstaller.TrySelectWheel(
            IndexJsonWithRequiresPython(">=3.10; platform_system == 'iOS'", IndexFile(
                "probe-1.0-py3-none-any.whl", "bdist_wheel", "aaa")),
            out _, out string? reason));

        Assert.Contains("cannot safely evaluate", reason!);
        Assert.Contains("unsupported release component", reason!);
    }

    [Theory]
    [InlineData("rich", "rich", null)]
    [InlineData("rich==13.9.4", "rich", "13.9.4")]
    [InlineData("et_xmlfile", "et_xmlfile", null)]
    public void PackageSpecsSplitIntoANameAndAnOptionalVersion(string spec, string name, string? version)
    {
        Assert.True(WheelInstaller.TrySplit(spec, out string parsed, out string? parsedVersion, out _));
        Assert.Equal(name, parsed);
        Assert.Equal(version, parsedVersion);
    }

    [Theory]
    [InlineData("../../etc/passwd")]
    [InlineData("rich; rm -rf /")]
    [InlineData("-rrequirements.txt")]
    [InlineData("rich==")]
    public void APackageSpecThatIsNotANameIsRefused(string spec)
    {
        Assert.False(WheelInstaller.TrySplit(spec, out _, out _, out string? error));
        Assert.NotNull(error);
    }

    [Fact]
    public void AWheelMemberThatClimbsOutOfTheTargetIsRefused()
    {
        using var archive = Archive(("rich/__init__.py", "ok"), ("../escaped.py", "not ok"));
        Assert.False(WheelInstaller.TryExtract(archive, _packages, Confined(_packages), out int files, out string? refusal));

        Assert.Equal(0, files);
        Assert.Contains("../escaped.py", refusal!);
        Assert.Contains("points outside", refusal!);
        Assert.False(System.IO.File.Exists(Path.Combine(_root, "escaped.py")));
    }

    [Fact]
    public void AWheelMemberWithAnAbsolutePathIsRefused()
    {
        using var archive = Archive(("/etc/cron.d/evil", "not ok"));
        Assert.False(WheelInstaller.TryExtract(archive, _packages, Confined(_packages), out _, out string? refusal));
        Assert.Contains("points outside", refusal!);
    }

    [SkippableFact]
    public void AWheelMemberThatWouldLandThroughASymlinkIsRefused()
    {
        // The archive name is innocent; a symlink already in the target is what
        // moves it. Only re-resolving each member catches this.
        TestPlatforms.CreateDirectorySymlink(Path.Combine(_packages, "rich"), _outside);
        using var archive = Archive(("rich/__init__.py", "not ok"));

        Assert.False(WheelInstaller.TryExtract(archive, _packages, Confined(_packages), out _, out string? refusal));
        Assert.Contains("outside", refusal!);
        Assert.False(System.IO.File.Exists(Path.Combine(_outside, "__init__.py")));
    }

    [Fact]
    public async Task AnExpandedWheelIsRejectedBeforeWritingAnyMember()
    {
        using var archive = Archive(("probe/one.py", "1234"), ("probe/two.py", "56"));

        (bool ok, int files, string? refusal) = await WheelInstaller.TryExtractAsync(
            archive, _packages, Confined(_packages), maxExpandedBytes: 5, maxFiles: 10,
            CancellationToken.None);

        Assert.False(ok);
        Assert.Equal(0, files);
        Assert.Contains("expanded wheel exceeds the 5 bytes limit", refusal!);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task AFileHeavyWheelIsRejectedBeforeWritingAnyMember()
    {
        using var archive = Archive(("probe/one.py", "1"), ("probe/two.py", "2"), ("probe/three.py", "3"));

        (bool ok, int files, string? refusal) = await WheelInstaller.TryExtractAsync(
            archive, _packages, Confined(_packages), maxExpandedBytes: 100, maxFiles: 2,
            CancellationToken.None);

        Assert.False(ok);
        Assert.Equal(0, files);
        Assert.Contains("more than 2 archive entries", refusal!);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task DirectoryEntriesCannotBypassTheArchiveCountLimit()
    {
        using var archive = Archive(("one/", ""), ("two/", ""), ("three/", ""));

        (bool ok, int files, string? refusal) = await WheelInstaller.TryExtractAsync(
            archive, _packages, Confined(_packages), maxExpandedBytes: 100, maxFiles: 2,
            CancellationToken.None);

        Assert.False(ok);
        Assert.Equal(0, files);
        Assert.Contains("more than 2 archive entries", refusal!);
        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public async Task ExtractionObservesCancellationBeforeWriting()
    {
        using var archive = Archive(("probe/__init__.py", "answer = 42\n"));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => WheelInstaller.TryExtractAsync(
            archive, _packages, Confined(_packages), WheelInstaller.MaxExpandedBytes,
            WheelInstaller.MaxArchiveFiles, cancellation.Token));

        Assert.Empty(Directory.GetFileSystemEntries(_packages));
    }

    [Fact]
    public void APureWheelUnpacksIntoTheTargetDirectory()
    {
        using var archive = Archive(
            ("rich/__init__.py", "VERSION = '13.9.4'"),
            ("rich/console.py", "class Console: pass"),
            ("rich-13.9.4.dist-info/METADATA", "Name: rich"));

        Assert.True(WheelInstaller.TryExtract(archive, _packages, Confined(_packages), out int files, out string? refusal));
        Assert.Null(refusal);
        Assert.Equal(3, files);
        Assert.Equal("VERSION = '13.9.4'", System.IO.File.ReadAllText(Path.Combine(_packages, "rich", "__init__.py")));
        // pip list reads the dist-info directories, so the layout has to be this one.
        Assert.True(Directory.Exists(Path.Combine(_packages, "rich-13.9.4.dist-info")));
    }

    // =====================================================================================
    // with a real interpreter
    // =====================================================================================

    [LivePythonFact]
    public async Task ALiveInterpreterPrintsAndStreamsEachLine()
    {
        EmbeddedPython python = Live();
        var lines = new List<string>();
        InterpreterContext context = Context() with { OnStdoutLine = lines.Add };

        ExecutionResult result = await python.RunCodeAsync("print('hello'); print('world')", [], context, CancellationToken.None);

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("hello\nworld\n", result.Stdout);
        Assert.Equal(["hello", "world"], lines);
    }

    [LivePythonFact]
    public void ALiveInterpreterReportsItsVersion()
    {
        EmbeddedPython python = Live();
        Assert.True(python.IsAvailable, python.UnavailableReason);
        Assert.StartsWith("3.", python.Version);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterSetsArgvAndMainTheWayTheRealOneDoes()
    {
        EmbeddedPython python = Live();
        ExecutionResult result = await python.RunCodeAsync(
            "import sys; print(sys.argv, __name__)", ["a", "b"], Context(), CancellationToken.None);

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("['-c', 'a', 'b'] __main__\n", result.Stdout);
    }

    [LivePythonFact]
    public async Task EachRunGivesBlockingSocketsItsOwnFiniteStallTimeout()
    {
        EmbeddedPython python = Live();
        ExecutionPolicy quick = Policy(network: true) with
        {
            DefaultTimeout = TimeSpan.FromMilliseconds(250),
        };
        ExecutionPolicy patient = quick with
        {
            DefaultTimeout = TimeSpan.FromSeconds(2),
        };

        ExecutionResult first = await python.RunCodeAsync(
            "import socket; print(socket.getdefaulttimeout())", [], Context(quick), CancellationToken.None);
        ExecutionResult second = await python.RunCodeAsync(
            "import socket; print(socket.getdefaulttimeout())", [], Context(patient), CancellationToken.None);

        Assert.True(first.Ok, first.Stderr);
        Assert.Equal("0.25\n", first.Stdout);
        Assert.True(second.Ok, second.Stderr);
        Assert.Equal("2.0\n", second.Stdout);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterRunsAScriptWithItsOwnDirectoryOnThePath()
    {
        EmbeddedPython python = Live();
        System.IO.File.WriteAllText(Path.Combine(_work, "helper.py"), "VALUE = 41\n");
        string script = Path.Combine(_work, "main.py");
        System.IO.File.WriteAllText(script, "import helper, sys\nprint(helper.VALUE + 1, __name__, sys.argv[0])\n");

        ExecutionResult result = await python.RunScriptAsync(script, ["main.py"], Context(), CancellationToken.None);

        Assert.Equal(0, result.ExitCode);
        Assert.Equal("42 __main__ main.py\n", result.Stdout);
    }

    [LivePythonFact]
    public async Task SessionModulesAndPinnedReplacementsAreIsolatedWhileStdlibStaysWarm()
    {
        EmbeddedPython python = Live();
        string moduleName = "tensoragent_isolation_" + Guid.NewGuid().ToString("N");
        string cacheMarker = "_tensoragent_cache_" + Guid.NewGuid().ToString("N");

        InterpreterContext Session(string name, string value)
        {
            string root = Path.Combine(_root, name);
            string work = Path.Combine(root, "work");
            string temp = Path.Combine(root, "tmp");
            string packages = Path.Combine(root, "packages");
            Directory.CreateDirectory(work);
            Directory.CreateDirectory(temp);
            Directory.CreateDirectory(packages);
            File.WriteAllText(Path.Combine(packages, moduleName + ".py"), $"VALUE = '{value}'\n");
            var policy = new ExecutionPolicy(
                AllowScripts: true,
                AllowNetwork: false,
                WorkRoot: work,
                ReadableRoots: [],
                TempRoot: temp)
            {
                PackageRoot = packages,
                DefaultTimeout = TimeSpan.FromSeconds(20),
            };
            return new InterpreterContext(
                work,
                new Dictionary<string, string>
                {
                    ["HOME"] = work,
                    ["TMPDIR"] = temp,
                    ["PYTHONPATH"] = packages,
                },
                policy);
        }

        InterpreterContext alpha = Session("alpha", "alpha");
        InterpreterContext bravo = Session("bravo", "bravo");
        ExecutionResult first = await python.RunCodeAsync(
            $"import colorsys, {moduleName}\n"
                + $"setattr(colorsys, '{cacheMarker}', 'warm')\n"
                + $"print({moduleName}.VALUE)\n",
            [], alpha, CancellationToken.None);
        ExecutionResult second = await python.RunCodeAsync(
            $"import colorsys, {moduleName}\n"
                + $"print({moduleName}.VALUE, getattr(colorsys, '{cacheMarker}', 'cold'))\n",
            [], bravo, CancellationToken.None);

        string bravoModule = Path.Combine(bravo.Policy.PackageRoot!, moduleName + ".py");
        File.WriteAllText(bravoModule, "VALUE = 'bravo-replaced'\n");
        ExecutionResult replaced = await python.RunCodeAsync(
            $"import colorsys, {moduleName}\n"
                + $"print({moduleName}.VALUE, getattr(colorsys, '{cacheMarker}', 'cold'))\n"
                + $"delattr(colorsys, '{cacheMarker}')\n",
            [], bravo, CancellationToken.None);

        Assert.True(first.Ok, first.Stderr);
        Assert.Equal("alpha\n", first.Stdout);
        Assert.True(second.Ok, second.Stderr);
        Assert.Equal("bravo warm\n", second.Stdout);
        Assert.True(replaced.Ok, replaced.Stderr);
        Assert.Equal("bravo-replaced warm\n", replaced.Stdout);
    }

    [LiveStagedPythonFact]
    public async Task SessionAndWorkingDirectoryOverrideAnAlreadyCachedBundledPackage()
    {
        EmbeddedPython python = Live();

        // Warm the copy staged with the app before the session has its own. This is
        // the case sys.path ordering alone cannot fix: CPython normally returns the
        // existing sys.modules entry without looking at any path.
        ExecutionResult warmed = await python.RunCodeAsync(
            "import certifi; print('bundled', certifi.__file__)",
            [], Context(), CancellationToken.None);
        Assert.True(warmed.Ok, warmed.Stderr);
        Assert.StartsWith("bundled ", warmed.Stdout, StringComparison.Ordinal);

        string sessionPackage = Path.Combine(_packages, "certifi");
        Directory.CreateDirectory(sessionPackage);
        File.WriteAllText(
            Path.Combine(sessionPackage, "__init__.py"),
            "SOURCE = 'session'\n");

        ExecutionResult fromSession = await python.RunCodeAsync(
            "import certifi; print(certifi.SOURCE, certifi.__file__)",
            [], Context(), CancellationToken.None);
        Assert.True(fromSession.Ok, fromSession.Stderr);
        Assert.Equal(
            $"session {Real(Path.Combine(sessionPackage, "__init__.py"))}\n",
            fromSession.Stdout);

        string workingPackage = Path.Combine(_work, "certifi");
        Directory.CreateDirectory(workingPackage);
        File.WriteAllText(
            Path.Combine(workingPackage, "__init__.py"),
            "SOURCE = 'working'\n");

        ExecutionResult fromWorkingDirectory = await python.RunCodeAsync(
            "import certifi; print(certifi.SOURCE, certifi.__file__)",
            [], Context(), CancellationToken.None);
        Assert.True(fromWorkingDirectory.Ok, fromWorkingDirectory.Stderr);
        Assert.Equal(
            $"working {Path.GetFullPath(Path.Combine(workingPackage, "__init__.py"))}\n",
            fromWorkingDirectory.Stdout);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterHonoursSystemExit()
    {
        EmbeddedPython python = Live();
        Assert.Equal(3, (await python.RunCodeAsync("raise SystemExit(3)", [], Context(), CancellationToken.None)).ExitCode);
        Assert.Equal(0, (await python.RunCodeAsync("raise SystemExit(0)", [], Context(), CancellationToken.None)).ExitCode);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterPrintsARealTracebackAndFails()
    {
        EmbeddedPython python = Live();
        ExecutionResult result = await python.RunCodeAsync(
            "def f():\n    raise ValueError('boom')\nf()", [], Context(), CancellationToken.None);

        Assert.Equal(1, result.ExitCode);
        Assert.Contains("Traceback (most recent call last):", result.Stderr);
        Assert.Contains("ValueError: boom", result.Stderr);
        // The bootstrap's own frame must not be in it.
        Assert.DoesNotContain("_run", result.Stderr);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterRefusesToWriteOutsideTheSandbox()
    {
        EmbeddedPython python = Live();
        string escape = Path.Combine(_outside, "bad.txt").Replace("\\", "\\\\");
        ExecutionResult result = await python.RunCodeAsync(
            $"open('{escape}', 'w').write('x')", [], Context(), CancellationToken.None);

        Assert.Equal(1, result.ExitCode);
        Assert.Contains("PermissionError", result.Stderr);
        Assert.Contains("Permission denied", result.Stderr);
        Assert.False(System.IO.File.Exists(Path.Combine(_outside, "bad.txt")));
    }

    [LivePythonFact]
    public async Task ALiveInterpreterRefusesTheNetworkWithTheSharedWording()
    {
        EmbeddedPython python = Live();
        ExecutionResult result = await python.RunCodeAsync(
            "import socket; socket.socket()", [], Context(), CancellationToken.None);

        Assert.Equal(1, result.ExitCode);
        Assert.Contains(ExecutionPolicy.NetworkDisabledMessage, result.Stderr);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterRefusesToStartAProcess()
    {
        EmbeddedPython python = Live();
        ExecutionResult result = await python.RunCodeAsync(
            "import os; os.system('echo hi')", [], Context(), CancellationToken.None);

        Assert.Equal(1, result.ExitCode);
        Assert.Contains(ExecutionPolicy.ProcessesUnavailableMessage, result.Stderr);
    }

    [LivePythonFact]
    public async Task ALiveInterpreterStopsARunThatOutlivesItsDeadline()
    {
        EmbeddedPython python = Live();
        InterpreterContext context = Context() with { Timeout = TimeSpan.FromSeconds(2) };

        ExecutionResult result = await python.RunCodeAsync("while True:\n    pass\n", [], context, CancellationToken.None);

        Assert.True(result.TimedOut);
        Assert.Equal(ExecutionResult.TimeoutExitCode, result.ExitCode);
    }

    [LivePythonFact]
    public async Task ALiveSyntaxCheckReportsPathAndLine()
    {
        EmbeddedPython python = Live();
        string script = Path.Combine(_work, "bad.py");
        System.IO.File.WriteAllText(script, "x = 1\ndef (:\n");

        SyntaxCheckResult result = await python.CheckSyntaxAsync(script, CancellationToken.None);

        Assert.False(result.Ok);
        Assert.StartsWith($"{script}:2:", result.Message);
    }

    [LivePythonFact]
    public async Task ALiveSyntaxCheckPassesGoodSource()
    {
        EmbeddedPython python = Live();
        string script = Path.Combine(_work, "good.py");
        System.IO.File.WriteAllText(script, "def f(x):\n    return x + 1\n");

        Assert.True((await python.CheckSyntaxAsync(script, CancellationToken.None)).Ok);
    }

    // =====================================================================================
    // helpers
    // =====================================================================================

    private static EmbeddedPython Live()
        => new(Environment.GetEnvironmentVariable(LivePythonFactAttribute.RootVariable));

    private ConfinedPaths Confined(string target) => new(Policy() with { WorkRoot = target });

    private static string Between(string text, string start, string end)
    {
        int from = text.IndexOf(start, StringComparison.Ordinal);
        Assert.True(from >= 0, $"'{start}' is not in the generated source");
        from += start.Length;
        int to = text.IndexOf(end, from, StringComparison.Ordinal);
        Assert.True(to >= 0, $"'{end}' does not close '{start}'");
        return text[from..to];
    }

    private static ZipArchive Archive(params (string Name, string Content)[] members)
    {
        var buffer = new MemoryStream();
        using (var writing = new ZipArchive(buffer, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach ((string name, string content) in members)
            {
                ZipArchiveEntry entry = writing.CreateEntry(name);
                using Stream stream = entry.Open();
                stream.Write(Encoding.UTF8.GetBytes(content));
            }
        }
        buffer.Position = 0;
        return new ZipArchive(buffer, ZipArchiveMode.Read);
    }

    private static byte[] WheelBytes(params (string Name, string Content)[] members)
    {
        using var buffer = new MemoryStream();
        using (var archive = new ZipArchive(buffer, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach ((string name, string content) in members)
            {
                ZipArchiveEntry entry = archive.CreateEntry(name);
                using Stream stream = entry.Open();
                stream.Write(Encoding.UTF8.GetBytes(content));
            }
        }
        return buffer.ToArray();
    }

    private static string IndexFile(
        string name,
        string type,
        string sha256,
        bool yanked = false,
        string? url = null,
        string? requiresPython = null)
        => $$"""
            {
              "filename": "{{name}}",
              "url": "{{url ?? $"https://files.pythonhosted.org/packages/{name}"}}",
              "packagetype": "{{type}}",
              "digests": { "sha256": "{{sha256}}" },
              "yanked": {{(yanked ? "true" : "false")}},
              "requires_python": {{(requiresPython is null ? "null" : JsonSerializer.Serialize(requiresPython))}}
            }
            """;

    private static string IndexJson(params string[] files)
        => $$"""{ "info": { "name": "x" }, "urls": [ {{string.Join(",", files)}} ] }""";

    private static string IndexJsonWithRequiresPython(string requiresPython, params string[] files)
        => $$"""{ "info": { "name": "x", "requires_python": {{JsonSerializer.Serialize(requiresPython)}} }, "urls": [ {{string.Join(",", files)}} ] }""";

    private static string IndexJsonWithDependencies(string[] dependencies, params string[] files)
        => $$"""{ "info": { "name": "x", "requires_dist": {{JsonSerializer.Serialize(dependencies)}} }, "urls": [ {{string.Join(",", files)}} ] }""";

    private static HttpResponseMessage TextResponse(HttpStatusCode status, string content) => new(status)
    {
        Content = new StringContent(content, Encoding.UTF8, "application/json"),
    };

    private static HttpResponseMessage BytesResponse(byte[] content) => new(HttpStatusCode.OK)
    {
        Content = new ByteArrayContent(content),
    };

    private static HttpResponseMessage DeclaredLengthResponse(long length)
    {
        var response = BytesResponse([]);
        response.Content.Headers.ContentLength = length;
        return response;
    }

    private static HttpResponseMessage StreamResponse(Stream stream) => new(HttpStatusCode.OK)
    {
        Content = new StreamContent(stream),
    };

    private static HttpResponseMessage RedirectResponse(string location)
    {
        var response = new HttpResponseMessage(HttpStatusCode.Found);
        response.Headers.Location = new Uri(location);
        return response;
    }

    private sealed class RecordingHttpHandler(
        Func<HttpRequestMessage, HttpResponseMessage> responseFor) : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = new();

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Requests.Add(request.RequestUri!);
            return Task.FromResult(responseFor(request));
        }
    }

    private sealed class CancelOnFirstReadStream(
        byte[] bytes, CancellationTokenSource cancellation) : MemoryStream(bytes, writable: false)
    {
        private bool _cancelled;

        public override ValueTask<int> ReadAsync(
            Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            int read = Read(buffer.Span);
            if (!_cancelled)
            {
                _cancelled = true;
                cancellation.Cancel();
            }
            return ValueTask.FromResult(read);
        }
    }
}
