// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.Extensions.Logging;
using TensorAgent.Maui.Hosting;
using TensorSharp.Models.Media;
#if IOS || MACCATALYST
using TensorSharp.Models.Media.Apple;
#endif

namespace TensorAgent.Maui;

public static class MauiProgram
{
#if IOS
    /// <summary>
    /// The scheduler's own name for the solo-prefill chunk size. Left overridable so a
    /// device experiment can try another value without a rebuild, which is how 1024
    /// was chosen.
    /// </summary>
    private const string SoloPrefillChunkVariable = "TS_SCHED_SOLO_PREFILL_CHUNK";
    private const string PrefixCheckpointBudgetVariable = "TS_PREFIX_CHECKPOINTS_MAX";

    /// <summary>
    /// ggml-metal reads this at device init to decide whether to keep every Metal
    /// buffer in a residency set. The set is what pins the weights.
    /// </summary>
    private const string MetalNoResidencyVariable = "GGML_METAL_NO_RESIDENCY";

    // Native, not Environment.SetEnvironmentVariable: on this runtime the managed
    // call updates a managed copy that a C getenv never sees.
    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern int setenv(string name, string value, int overwrite);

    [System.Runtime.InteropServices.DllImport("libSystem.dylib")]
    private static extern int unsetenv(string name);
#endif

    public static MauiApp CreateMauiApp()
    {
        // Before the host exists: it picks the interface language as it loads the
        // settings, and with no choice saved that is the first of these it has strings for.
        Core.Localization.Loc.SystemLanguages = Services.SystemLanguages.Preferred;

#if IOS
        ApplyPhoneEngineSettings();
#else
        // A desktop app started from the Finder or the Dock inherits launchd's PATH,
        // not the one a terminal has, so node, npm and a Homebrew python3 are simply
        // not found and the skills that need them fail on their first step. The login
        // shell is asked once, here, before anything starts a process. See
        // DesktopEnvironment; a no-op on Windows, where PATH comes from the registry.
        if (Core.Hosting.DesktopEnvironment.ImportLoginShellPath() is { } path)
            Console.WriteLine($"TensorAgent: PATH from the login shell ({path.Split(Path.PathSeparator).Length} entries)");
        ApplyDesktopEngineSettings();
#endif

#if IOS || MACCATALYST
        // Media before anything else can decode: a photo from Photos is HEIC, a clip from
        // the camera roll is H.264 and a Voice Memo is .m4a, and MediaCodecs starts on
        // managed defaults that read none of those — they throw a "register a platform
        // provider" NotSupportedException instead. TensorSharp.Models has a module
        // initializer that does this when its assembly loads, but that is a side effect of
        // something else happening first; calling it here makes the app's dependency on
        // ImageIO/AVFoundation visible where the app is assembled, and it is idempotent.
        AppleMediaProvider.Register();
#endif
        Console.WriteLine($"TensorAgent: media providers {MediaCodecs.Describe()}");
#if DEBUG && (IOS || MACCATALYST)
        // Debug only, and the only place the Apple media provider is ever executed: the repo's
        // xunit suite is a net10.0 host that cannot load an iOS assembly, so ImageIO and
        // AVFoundation are exercised here against files this device encodes itself, and
        // scripts/verify-sim.sh fails the simulator run if any check reports false. Costs a
        // fraction of a second at launch and nothing at all in a Release build.
        Console.WriteLine("TensorAgent: media probe " + System.Text.Json.JsonSerializer.Serialize(
            MediaProbe.Run(), new System.Text.Json.JsonSerializerOptions(System.Text.Json.JsonSerializerDefaults.Web)));
#endif

        MauiAppBuilder builder = MauiApp.CreateBuilder();
        builder.UseMauiApp<App>();

        // The Web UI is the app's own page, TensorAgent.Maui/wwwroot, linked into the
        // bundle as webui/ (see the BundleResource items in the csproj).
        string webRoot = Path.Combine(AppBundle.ResourceDirectory, "webui");
        // Console logging is what `simctl launch --console` shows and what a device log
        // capture picks up -- while something is attached to it. A device console
        // detaches the moment the app is backgrounded, which is when the failures worth
        // reading about happen, so warnings and errors are ALSO written to a file that
        // comes back off the phone afterwards. See DurableErrorLog.
        builder.Logging.AddConsole();
        builder.Logging.AddProvider(new Core.Hosting.DurableErrorLog(
            Hosting.LoopbackWebHost.DeviceLogsDirectory()));
        builder.Services.AddSingleton(sp => new LoopbackWebHost(
            webRoot, sp.GetService<ILoggerFactory>()));
        builder.Services.AddSingleton<MainPage>();
        builder.Services.AddSingleton<Pages.SessionsPage>();
        builder.Services.AddSingleton<Pages.ModelsPage>();
        builder.Services.AddSingleton<Pages.SettingsPage>();
        builder.Services.AddSingleton<Pages.AboutPage>();
        builder.Services.AddSingleton<AppShell>();

        return builder.Build();
    }

#if !IOS
    /// <summary>
    /// Qwen3.8 Flash Next reads its BF16 hyper-connection matrices as a Q8_0 copy made once at load
    /// (Strata's default read): 1.26 GB a token becomes 0.67 GB. The fastest presets turn it on
    /// (config/qwen3.8-flash-next.json, config/qwen3.8-flash-next-cuda.json), and every measurement of
    /// the catalog's Flash Next entries assumes it. Only the qwen4exp model reads it, once, when its
    /// type initializes, so it is set before any model loads. A value already in the environment wins.
    /// </summary>
    private const string FlashNextHcQ8Variable = "TS_Q4E_HC_Q8";

    private static void ApplyDesktopEngineSettings()
    {
        // Managed-to-managed (Qwen4ExpModel reads it with Environment.GetEnvironmentVariable), so
        // SetEnvironmentVariable is enough; no native getenv reads it.
        if (Environment.GetEnvironmentVariable(FlashNextHcQ8Variable) is not { Length: > 0 })
            Environment.SetEnvironmentVariable(FlashNextHcQ8Variable, "1");
        Console.WriteLine($"TensorAgent: Flash Next hyper-connection read {(Environment.GetEnvironmentVariable(FlashNextHcQ8Variable) == "1" ? "Q8_0" : "BF16")}");
    }
#endif

#if IOS
    /// <summary>
    /// The engine settings measured on an iPhone; a desktop sets only <see cref="ApplyDesktopEngineSettings"/>.
    /// </summary>
    private static void ApplyPhoneEngineSettings()
    {
        // No Metal residency set on the phone. The set keeps every buffer -- the
        // 4.9 GB of Qwen 9B weights included -- wired for the life of the model, so
        // a 12 GB phone sits at 7.0-7.1 GB wired for the whole of an agentic turn,
        // tool rounds included, when the GPU is idle for a minute at a time and the
        // weights could be reclaimed and re-faulted from flash. MEASURED on the
        // iPhone 17 Pro Max with the same prompt and settings, Debug build:
        //   residency set on:  killed (jetsam, no report) at a 10k-token context,
        //                      wired 7.0-7.1 GB throughout, the app itself at 0.96 GB
        //   residency set off: 23k+ tokens of context alive, wired 6.2-7.1 GB as
        //                      buffers come and go, decode within run-to-run noise
        //                      (6.8-7.7 vs 7.3-8.3 tok/s on the interpreter build)
        // The set exists so a Mac near its working-set limit does not thrash
        // re-requesting residency; a phone near its limit is killed instead, which
        // is the worse of the two. A value already in the environment (a devicectl
        // launch experimenting the other way) is respected.
        // ggml-metal tests the variable's PRESENCE, so "0" would still switch the set
        // off; here "0" means "keep the residency set" and is removed from the
        // environment, which is how a devicectl launch A/Bs the two on the phone.
        string? residency = Environment.GetEnvironmentVariable(MetalNoResidencyVariable);
        bool residencySetsOn;
        try
        {
            if (residency == "0")
            {
                unsetenv(MetalNoResidencyVariable);
                residencySetsOn = true;
            }
            else
            {
                if (residency is not { Length: > 0 })
                    setenv(MetalNoResidencyVariable, "1", 1);
                residencySetsOn = false;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"TensorAgent: could not set {MetalNoResidencyVariable}: {ex.Message}");
            residencySetsOn = residency is not { Length: > 0 };
        }
        Console.WriteLine($"TensorAgent: Metal residency sets {(residencySetsOn ? "on" : "off")} (ggml-metal reads {MetalNoResidencyVariable} at device init)");

        // Prefill in chunks a phone can actually hold.
        //
        // A solo request prefills up to min(SoloPrefillChunkSize, MaxNumBatchedTokens)
        // tokens in ONE fused pass -- 4096 by default. That default is written for a
        // desktop GPU, where a big chunk is several times faster than splitting one,
        // and it is fatal here: on an iPhone 17 Pro Max, pasting a 140-line document
        // into the chat got as far as "Expanded Gemma4 global attention cache to 8192
        // tokens" and then the app was killed outright -- "App terminated due to
        // signal 9", jetsam, with no answer and no error the user could see.
        //
        // Measured on that device with the same 22 kB paste and gemma-4-E2B on Metal:
        //   default (4096) -> killed by jetsam, no answer
        //   2048           -> survives, correct answer, 50.0 s
        //   1024           -> survives, correct answer, 42.4 s
        //
        // 1024 is not a reluctant compromise: it was FASTER than 2048 here, because on
        // a memory-constrained device the pressure a big chunk creates costs more than
        // the fused pass saves. The desktop reasoning does not transfer, so the phone
        // gets its own value rather than the shared default.
        //
        // Set through the environment because that is the seam SchedulerConfig already
        // reads, and it is read when the engine is constructed -- which happens after
        // this. Managed-to-managed, so SetEnvironmentVariable is enough here (a NATIVE
        // getenv would not see it).
        if (Environment.GetEnvironmentVariable(SoloPrefillChunkVariable) is not { Length: > 0 })
            Environment.SetEnvironmentVariable(SoloPrefillChunkVariable, "1024");
        Console.WriteLine($"TensorAgent: solo prefill chunk {Environment.GetEnvironmentVariable(SoloPrefillChunkVariable)} tokens");

        // One shared-prefix checkpoint, not the engine's two. A checkpoint is a whole
        // copy of the model's state at the end of the system prompt -- some 45 MB for
        // Gemma 4 E2B, a few hundred for a 9B or 12B -- kept so every new chat starts
        // from it instead of re-prefilling it (see IBatchedPagedModel.SupportsPrefixCheckpoints).
        // The engine keeps two so a thinking toggle, which changes the prefix on Gemma 4,
        // has both ready; next to a jetsam limit one is the right trade, and the second
        // mode simply re-prefills once and takes the checkpoint over.
        if (Environment.GetEnvironmentVariable(PrefixCheckpointBudgetVariable) is not { Length: > 0 })
            Environment.SetEnvironmentVariable(PrefixCheckpointBudgetVariable, "1");
        Console.WriteLine($"TensorAgent: shared-prefix checkpoints kept {Environment.GetEnvironmentVariable(PrefixCheckpointBudgetVariable)}");
    }
#endif
}
