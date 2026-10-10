using System;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Settings;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Hands the engine the memory budget a phone actually has, before a model loads.
///
/// <para>
/// The catalog has always carried a per-entry <see cref="CatalogModel.ContextLength"/>
/// and <see cref="CatalogModel.KvCacheDtype"/>, and until this existed neither reached
/// the engine. The context length's only reader was the JSON payload the page renders;
/// <see cref="AppSettings.ContextLength"/>, documented as "override of the catalog
/// entry's context length", was read by nothing at all; and
/// <c>KvCacheDtypeConfig.ConfigureFromEnvironment()</c> was called by TensorSharp.Server
/// and TensorSharp.Cli but never by the MAUI head. So the engine fell back to the GGUF's
/// own number, and for Qwen3.5 9B that is 262,144.
/// </para>
///
/// <para>
/// That is not a theoretical ceiling. Qwen3.5 9B spends 2 (K+V) x 8 attention layers x 4
/// KV heads x 256 head dim x F16 = 32 KiB of KV per token, and on Metal it is charged
/// TWICE: once for the host tensor (an anonymous mmap from GgmlMemoryPool) and once for
/// the Metal side, which refuses the zero-copy wrap for read-write tensors and allocates
/// its own buffer -- posix_memalign on iOS. 64 KiB per token against a jetsam budget of
/// roughly two thirds of the device's RAM. MEASURED on this model, backend ggml_metal,
/// with a 24,696-token prompt (a pasted document -- the ordinary case this is for):
///   no MAX_CONTEXT: cache expanded to 32768 tokens, peak footprint 5,679 MB
///   MAX_CONTEXT=8192:                                peak footprint   943 MB
/// The first number is what killed TensorAgent on a 12 GB iPhone.
/// </para>
///
/// <para>
/// Setting MAX_CONTEXT also switches the engine from growing the cache on demand to
/// reserving the whole window at load (ModelBase.ResolveInitialCacheAllocationLength
/// skips its GPU cap when the context is explicit). That is the behaviour to want here:
/// the reservation is bounded and paid once, instead of arriving as a geometric growth
/// mid-conversation whose superseded blocks GgmlMemoryPool then retains. A budget that
/// is wrong is discovered at load, where it can be reported, rather than as a kill.
/// </para>
///
/// <para>
/// A prompt longer than the window is not an error: ChatGenerationPipeline's
/// TruncatePromptToContext trims history to fit and reports it, which is what a chat
/// should do regardless of the device.
/// </para>
/// </summary>
public static class EngineMemoryPolicy
{
    /// <summary>The engine reads this at model construction (ModelBase.ResolveConfiguredContextLength).</summary>
    public const string MaxContextVariable = "MAX_CONTEXT";

    /// <summary>Read by KvCacheDtypeConfig.ConfigureFromEnvironment().</summary>
    public const string KvCacheDtypeVariable = "KV_CACHE_DTYPE";

    /// <summary>
    /// What the engine keeps beyond the one cache a turn is using, and how much of the
    /// window it commits before a request says what it needs. Each is an engine knob
    /// (<c>ExecutionOptions</c>) with a desktop default written for a machine with
    /// memory to spare; these are the phone's values, and every one is a measured
    /// number, not a guess.
    ///
    /// <para>
    /// The jetsam reports the phone kept tell the whole story. Every kill was a
    /// system-wide page shortage with 7.5-10 GB of the 12 GB wired while TensorAgent's
    /// own footprint was 2.3-5.6 GB: the weights are wired by Metal outside the
    /// footprint (see <see cref="ProcessMemory"/>), so the app has roughly 12 GB minus
    /// the weights minus ~3 GB of kernel and system to live in -- about 3.5 GB for a
    /// 5 GB model -- and the K/V cache is paid TWICE in it, host copy and Metal mirror.
    /// The phone's settings had the reply length at 262,144 tokens, so every request
    /// reserved the entire 32k window for its holder; the engine then kept up to four
    /// finished conversations' holders and parked up to 64 more, every one sized to
    /// the whole window, and the primary cache the engine loads with (also the whole
    /// window) sat idle behind them once the per-request path took over.
    /// </para>
    ///
    /// <para>
    /// The values: a cache starts at 2,048 tokens and grows as the conversation does
    /// (four doublings to reach 32k, each a copy of what is resident -- measured under
    /// a second in total on the phone); a request pre-reserves at most 1,024 tokens of
    /// reply beyond its prompt and grows on demand past that; one finished
    /// conversation stays resident for the follow-up turn (the shared-prefix
    /// checkpoint, kept separately, is what makes a NEW chat fast); and nothing is
    /// parked. Measured on the Mac with the phone's settings and the research-then-
    /// pptx prompt that was killing the app, 27 tool rounds to a 29k-token context:
    /// the host held 3.7 GB before, and the reduction is reported by the bench
    /// (benchmarks/TensorAgentTtftBench --scenarios agentic).
    /// </para>
    /// </summary>
    public const string KvInitialTokensVariable = "TS_KV_INITIAL_TOKENS";
    public const string KvGenerationReserveMaxVariable = "TS_KV_GENERATION_RESERVE_MAX";
    public const string KvHolderPoolMaxVariable = "TS_KV_HOLDER_POOL_MAX";
    public const string RetainedFusedCacheMaxVariable = "TS_RETAINED_FUSED_CACHE_MAX";

    public const int KvInitialTokens = 2048;
    public const int KvGenerationReserveMax = 1024;
    public const int KvHolderPoolMax = 0;
    public const int RetainedFusedCacheMax = 1;

    /// <summary>
    /// The K/V cache precisions the Settings screen offers, widest first.
    ///
    /// <para>
    /// The spellings are the engine's own (<c>KvCacheDtypeConfig.TryParse</c>), so what
    /// is stored in the settings file is what the environment variable carries and what
    /// the catalog entries are written in — one vocabulary, not three that have to be
    /// translated between.
    /// </para>
    /// </summary>
    public static readonly string[] KvCacheDtypes = { "f16", "q8_0", "q4_0" };

    /// <summary>
    /// The cache precision this load should ask for.
    ///
    /// <para>
    /// The user's setting wins, because it is the one the user can see. The catalog
    /// entry is the fallback for a settings file written before the setting existed, or
    /// carrying a value this build does not know — an unrecognised string must not
    /// reach the engine, where an unparseable value is silently ignored and leaves
    /// whatever the PREVIOUS model set still in force.
    /// </para>
    /// <para>
    /// Asking is all this does. A family that cannot read a block-quantized cache
    /// refuses it during model construction and substitutes f16
    /// (<c>ModelBase.RefuseUnsupportedBlockQuantizedKvCache</c>), which is why a global
    /// q4_0 is safe to default to even though half this catalog is Gemma 4.
    /// </para>
    /// </summary>
    public static string ResolveKvCacheDtype(CatalogModel model, AppSettings? settings)
    {
        ArgumentNullException.ThrowIfNull(model);
        string? chosen = settings?.KvCacheDtype?.Trim();
        if (!string.IsNullOrEmpty(chosen))
        {
            foreach (string known in KvCacheDtypes)
            {
                if (string.Equals(known, chosen, StringComparison.OrdinalIgnoreCase))
                    return known;
            }
        }
        return model.KvCacheDtype;
    }

    /// <summary>
    /// The context window <paramref name="model"/>'s entry gives a load on
    /// <paramref name="device"/> when the user has not chosen one: the phone's measured
    /// <see cref="CatalogModel.ContextLength"/>, or on a desktop the window the entry's tier
    /// affords (<see cref="CatalogModel.DesktopContextLength"/>). <see cref="Apply"/> loads
    /// with it and the catalog route reports it, so the two cannot disagree.
    /// </summary>
    public static int DefaultContextLength(CatalogModel model, DeviceClass device)
    {
        ArgumentNullException.ThrowIfNull(model);
        return device == DeviceClass.Desktop ? model.DesktopContextLength : model.ContextLength;
    }

    /// <summary>
    /// Apply <paramref name="model"/>'s budget for the load that is about to happen.
    /// Returns the context length handed to the engine, or 0 when the entry does not
    /// state one (the diffusion entries, which hold no KV cache) and the GGUF's own
    /// value is left alone.
    /// </summary>
    public static int Apply(CatalogModel model, AppSettings? settings, DeviceClass device = DeviceClass.Phone)
    {
        ArgumentNullException.ThrowIfNull(model);

        // The user's override wins where they set one; otherwise the catalog entry,
        // which is written per model against the device tier that is offered it: the
        // phone's measured window, or on a desktop the window the entry's tier affords
        // (CatalogModel.DesktopContextLength) -- the phone's 8,192 left a desktop chat
        // ~1k tokens beside TensorAgent's ~7.2k-token shared prompt, and every follow-up
        // compacted the conversation away. A LeanCaches entry's caches still start at
        // 2,048 tokens and grow, so the larger ceiling costs nothing until it is used.
        int context = settings?.ContextLength is int chosen && chosen > 0
            ? chosen
            : DefaultContextLength(model, device);

        if (context > 0)
            Environment.SetEnvironmentVariable(MaxContextVariable, context.ToString());
        else
            Environment.SetEnvironmentVariable(MaxContextVariable, null);

        string dtype = ResolveKvCacheDtype(model, settings);
        Environment.SetEnvironmentVariable(
            KvCacheDtypeVariable,
            string.IsNullOrWhiteSpace(dtype) ? null : dtype);

        // Managed-to-managed, and read when the model is constructed, which is after
        // this. The static config is what the model layer consults for the cache dtype;
        // without this call the variable above would be as inert as it was before.
        TensorSharp.Models.KvCacheDtypeConfig.ConfigureFromEnvironment();

        // What the engine may keep besides the cache in use, and how much it commits
        // ahead of a request. Read by the engine on every step and by the model at
        // construction, so setting them here -- before the load, on every load -- is
        // enough. See the summary on the constants for the numbers. A desktop host
        // clears them instead, so the engine's defaults apply and a phone budget set
        // earlier in the same process cannot outlive it.
        // The phone's budget, on a phone or for an entry too large to afford the desktop's.
        bool phone = device == DeviceClass.Phone || model.LeanCaches;
        Environment.SetEnvironmentVariable(KvInitialTokensVariable, phone ? KvInitialTokens.ToString() : null);
        Environment.SetEnvironmentVariable(KvGenerationReserveMaxVariable, phone ? KvGenerationReserveMax.ToString() : null);
        Environment.SetEnvironmentVariable(KvHolderPoolMaxVariable, phone ? KvHolderPoolMax.ToString() : null);
        Environment.SetEnvironmentVariable(RetainedFusedCacheMaxVariable, phone ? RetainedFusedCacheMax.ToString() : null);

        Console.WriteLine(
            $"TensorAgent: engine budget for {model.Id} -- context {(context > 0 ? context.ToString() : "from GGUF")}, " +
            $"KV cache {(string.IsNullOrWhiteSpace(dtype) ? "auto" : dtype)}"
            + (string.Equals(dtype, model.KvCacheDtype, StringComparison.OrdinalIgnoreCase)
                ? string.Empty
                : $" (setting; this entry asks for {model.KvCacheDtype})")
            + (phone
                ? $"; caches start at {KvInitialTokens} tokens, pre-reserve at most {KvGenerationReserveMax} of reply, "
                  + $"{RetainedFusedCacheMax} finished conversation kept, {KvHolderPoolMax} parked"
                : "; desktop budget: the engine's defaults size the caches and what is kept for reuse"));

        return context;
    }
}
