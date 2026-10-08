using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// The catalog's context length has to REACH the engine.
///
/// <para>
/// It did not, and that is what killed the app. Every entry carries a ContextLength,
/// and its only reader was the JSON the page renders; AppSettings.ContextLength,
/// documented as the user's override, was read by nothing at all. So the engine used
/// the GGUF's own number -- 262,144 for Qwen3.5 9B -- and a pasted document grew the
/// KV cache until jetsam killed the process. Measured on ggml_metal with a
/// 24,696-token prompt: 5,679 MB of physical footprint unbounded, 943 MB with the
/// budget applied.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class EngineMemoryPolicyTests : IDisposable
{
    private readonly string? _savedContext = Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable);
    private readonly string? _savedDtype = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable);
    private readonly string? _savedInitial = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable);
    private readonly string? _savedReserve = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable);
    private readonly string? _savedPool = Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable);
    private readonly string? _savedRetained = Environment.GetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable, _savedContext);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable, _savedDtype);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable, _savedInitial);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable, _savedReserve);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable, _savedPool);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable, _savedRetained);
    }

    private static CatalogModel Entry(string id) =>
        ModelCatalog.Find(id) ?? throw new InvalidOperationException($"catalog entry {id} is gone");

    [Fact]
    public void TheCatalogEntrysContextLengthReachesTheEngine()
    {
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        int applied = EngineMemoryPolicy.Apply(qwen, new AppSettings());

        Assert.Equal(qwen.ContextLength, applied);
        Assert.Equal(
            qwen.ContextLength.ToString(),
            Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable));
    }

    [Fact]
    public void ADesktopLoadLeavesTheKvKnobsToTheEngineEvenAfterAPhoneLoad()
    {
        // The phone's four knobs are the budget of a jetsam limit. The desktop app clears
        // them so the engine's own defaults apply -- and clears rather than skips, because a
        // value set by an earlier load in the same process would otherwise outlive it.
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        EngineMemoryPolicy.Apply(qwen, new AppSettings(), DeviceClass.Phone);
        Assert.NotNull(Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable));

        int context = EngineMemoryPolicy.Apply(qwen, AppSettings.DesktopDefaults(), DeviceClass.Desktop);

        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable));
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable));
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable));
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable));
        Assert.NotEqual(EngineMemoryPolicy.KvInitialTokens,
            TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment().KvInitialTokens);

        // What a desktop load still takes from the entry and the settings: the context the
        // catalog wrote and the K/V precision.
        Assert.Equal(qwen.ContextLength, context);
        Assert.Equal("q8_0", Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable));
    }

    /// <summary>
    /// A model that takes most of a Mac's memory keeps the phone's budget there too. With
    /// the engine's desktop defaults Qwen3.8 27B grew the Mac app to 18.8 GB beside its
    /// 17.6 GB of weights over seven conversations; the phone's budget held it at 8.8 GB
    /// with the same prompt reuse. Every other entry still leaves the knobs to the engine.
    /// </summary>
    [Fact]
    public void AnEntryThatAsksForLeanCachesKeepsThePhonesBudgetOnTheDesktop()
    {
        foreach (CatalogModel model in ModelCatalog.BuiltIn.Where(m => m.LeanCaches))
        {
            Assert.True(model.LeanCaches);
            EngineMemoryPolicy.Apply(model, AppSettings.DesktopDefaults(), DeviceClass.Desktop);

            var options = TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment();
            Assert.Equal(EngineMemoryPolicy.KvInitialTokens, options.KvInitialTokens);
            Assert.Equal(EngineMemoryPolicy.KvGenerationReserveMax, options.KvGenerationReserveMax);
            Assert.Equal(EngineMemoryPolicy.KvHolderPoolMax, options.KvHolderPoolMax);
            Assert.Equal(EngineMemoryPolicy.RetainedFusedCacheMax, options.RetainedFusedCacheBudget);
        }

        // And the next load of an ordinary entry clears them again.
        EngineMemoryPolicy.Apply(Entry("qwen3.5-9b-iq4xs"), AppSettings.DesktopDefaults(), DeviceClass.Desktop);
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable));
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable));
        foreach (string id in new[] { "muse-glimmer-30b-q4kxl", "qwen3.8-27b-q4kxl", "qwen3.8-flash-next-iq1m", "qwen3.8-flash-next-q2kxl" })
            Assert.True(Entry(id).LeanCaches);
        Assert.All(ModelCatalog.BuiltIn.Where(m => m.Experimental && m.MinDeviceMemoryGB >= 24),
            m => Assert.True(m.LeanCaches));
    }

    [Fact]
    public void TheIq1MFlashEntryAppliesItsBoundedContextAndF16CacheOnTheDesktop()
    {
        CatalogModel model = Entry("qwen3.8-flash-next-iq1m");
        // No user dtype override: exercise the entry's own precision rather than the
        // global desktop preference. The existing override tests cover that preference.
        AppSettings settings = AppSettings.DesktopDefaults();
        settings.KvCacheDtype = string.Empty;

        int context = EngineMemoryPolicy.Apply(model, settings, DeviceClass.Desktop);

        Assert.Equal(32768, context);
        Assert.Equal("32768", Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable));
        Assert.Equal("f16", Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable));
        var options = TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment();
        Assert.Equal(2048, options.KvInitialTokens);
        Assert.Equal(1024, options.KvGenerationReserveMax);
        Assert.Equal(1, options.RetainedFusedCacheBudget);
        Assert.Equal(0, options.KvHolderPoolMax);
    }

    [Fact]
    public void ThePhonesKvKnobsReachTheEngineOnEveryLoad()
    {
        // Each of these is an engine default written for a machine with memory to spare;
        // the phone's values are the ones EngineMemoryPolicy documents and measured.
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable, null);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable, null);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable, null);
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable, null);

        EngineMemoryPolicy.Apply(Entry("qwen3.5-9b-iq4xs"), new AppSettings());

        Assert.Equal(EngineMemoryPolicy.KvInitialTokens.ToString(),
            Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvInitialTokensVariable));
        Assert.Equal(EngineMemoryPolicy.KvGenerationReserveMax.ToString(),
            Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvGenerationReserveMaxVariable));
        Assert.Equal(EngineMemoryPolicy.KvHolderPoolMax.ToString(),
            Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvHolderPoolMaxVariable));
        Assert.Equal(EngineMemoryPolicy.RetainedFusedCacheMax.ToString(),
            Environment.GetEnvironmentVariable(EngineMemoryPolicy.RetainedFusedCacheMaxVariable));

        // And the engine reads exactly those spellings.
        var options = TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment();
        Assert.Equal(EngineMemoryPolicy.KvInitialTokens, options.KvInitialTokens);
        Assert.Equal(EngineMemoryPolicy.KvGenerationReserveMax, options.KvGenerationReserveMax);
        Assert.Equal(EngineMemoryPolicy.KvHolderPoolMax, options.KvHolderPoolMax);
        Assert.Equal(EngineMemoryPolicy.RetainedFusedCacheMax, options.RetainedFusedCacheBudget);
    }

    [Fact]
    public void TheUsersOwnOverrideWinsOverTheCatalog()
    {
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        int applied = EngineMemoryPolicy.Apply(qwen, new AppSettings { ContextLength = 4096 });

        Assert.Equal(4096, applied);
        Assert.Equal("4096", Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable));
    }

    [Fact]
    public void AnEntryThatStatesNoContextLeavesTheGgufsOwnValueAlone()
    {
        // Some architectures hold no KV cache and state ContextLength = 0. The reduced
        // built-in list currently has no such entry, so use a synthetic one to preserve
        // the engine-policy boundary without requiring an unrelated catalog card.
        Environment.SetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable, "8192");
        CatalogModel noContext = Entry("qwen3.5-9b-iq4xs") with
        {
            Id = "synthetic-no-context",
            ContextLength = 0,
        };

        Assert.Equal(0, EngineMemoryPolicy.Apply(noContext, new AppSettings()));
        Assert.Null(Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable));
    }

    [Fact]
    public void SwitchingModelsReplacesTheBudgetRatherThanKeepingTheOldOne()
    {
        EngineMemoryPolicy.Apply(Entry("qwen3.5-9b-iq4xs"), new AppSettings());
        string? first = Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable);

        EngineMemoryPolicy.Apply(Entry("gemma-4-e2b-q8"), new AppSettings());
        string? second = Environment.GetEnvironmentVariable(EngineMemoryPolicy.MaxContextVariable);

        // Against the catalog, not literals: the point of the test is that switching
        // REPLACES the budget, and hardcoding the numbers only breaks it when a
        // context is legitimately retuned.
        Assert.Equal(Entry("qwen3.5-9b-iq4xs").ContextLength.ToString(), first);
        Assert.Equal(Entry("gemma-4-e2b-q8").ContextLength.ToString(), second);
        Assert.NotNull(first);
        Assert.NotEqual(first, second);
    }

    /// <summary>
    /// With no preference of the user's, the entry's own dtype is what the engine gets.
    /// </summary>
    [Fact]
    public void TheKvCacheDtypeReachesTheEngineToo()
    {
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        EngineMemoryPolicy.Apply(qwen, new AppSettings { KvCacheDtype = string.Empty });

        Assert.Equal(qwen.KvCacheDtype, Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable));
    }

    /// <summary>
    /// The Settings screen's choice reaches the engine, and beats the catalog entry.
    ///
    /// <para>
    /// The catalog entry is per-model tuning; the setting is the user looking at their
    /// own phone and deciding they would rather spend precision than memory. Whichever
    /// way round one thinks about that, the thing a user can SEE has to be the thing
    /// that happens, or it is a switch that does nothing — which is the failure this
    /// whole file exists because of.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    [InlineData("q4_0")]
    public void TheChosenKvCacheDtypeBeatsTheCatalogEntry(string chosen)
    {
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        EngineMemoryPolicy.Apply(qwen, new AppSettings { KvCacheDtype = chosen });

        Assert.Equal(chosen, Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable));
    }

    /// <summary>
    /// A value this build does not recognise falls back to the entry rather than
    /// reaching the engine.
    ///
    /// <para>
    /// Not fussiness. KvCacheDtypeConfig.ConfigureFromEnvironment IGNORES a string it
    /// cannot parse, and its state is static and process-wide, so an unparseable value
    /// does not mean "the default" — it means whatever the PREVIOUS model set is still
    /// in force. A settings file from a newer build, or one edited by hand, would
    /// silently give the next model the last one's cache.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("q5_1")]
    [InlineData("nonsense")]
    [InlineData("  ")]
    public void AnUnknownKvCacheDtypeFallsBackToTheEntry(string bogus)
    {
        CatalogModel qwen = Entry("qwen3.5-9b-iq4xs");
        EngineMemoryPolicy.Apply(qwen, new AppSettings { KvCacheDtype = bogus });

        Assert.Equal(qwen.KvCacheDtype, Environment.GetEnvironmentVariable(EngineMemoryPolicy.KvCacheDtypeVariable));
    }

    /// <summary>
    /// The default is q4_0, and it is a value the engine can actually parse.
    ///
    /// <para>
    /// Both halves matter. The first is the product decision; the second is that every
    /// string this setting can hold has to survive KvCacheDtypeConfig.TryParse, because
    /// one that does not is inert in the silent way described above.
    /// </para>
    /// </summary>
    [Fact]
    public void TheDefaultCacheIsQ4AndEveryOfferedValueIsOneTheEngineParses()
    {
        Assert.Equal("q4_0", new AppSettings().KvCacheDtype);
        Assert.Contains("q4_0", EngineMemoryPolicy.KvCacheDtypes);

        foreach (string offered in EngineMemoryPolicy.KvCacheDtypes)
        {
            Assert.True(
                TensorSharp.Models.KvCacheDtypeConfig.TryParse(offered, out _),
                $"the Settings screen offers {offered}, which the engine cannot parse and would ignore");
        }
    }

    /// <summary>
    /// A catalog entry must not ask for a K/V dtype its family cannot read.
    ///
    /// <para>
    /// Gemma 4 declines a block-quantized cache (Gemma4Model.SupportsBlockQuantizedKvCache):
    /// its sliding-window layers use a CIRCULAR cache whose managed helpers are float-only,
    /// and an ordinary prompt can reach them. Setting q8_0 on a Gemma entry therefore
    /// does not merely fall back -- before the load-time refusal it crashed the app
    /// with "Requires a Float32 tensor, but found Q8_0" out of CopyToCacheCircular the moment
    /// a user typed. Qwen3.5 takes the fused graph, whose native side is dtype-generic,
    /// and keeps the memory win.
    /// </para>
    /// </summary>
    [Fact]
    public void NoEntryAsksForAKvDtypeItsFamilyCannotRead()
    {
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            bool blockQuant = m.KvCacheDtype is "q8_0" or "q4_0";
            if (m.Family == CatalogFamily.Gemma4)
            {
                Assert.False(blockQuant,
                    $"{m.Id} asks for {m.KvCacheDtype}, but Gemma 4 refuses a block-quantized cache; "
                    + "its circular sliding-window helpers are float-only and the managed path is "
                    + "reachable from a plain prompt");
            }
        }
    }

    /// <summary>
    /// The other direction: the families that CAN take it should still be taking it, so the
    /// memory win is not quietly reverted along with a Gemma fix.
    /// </summary>
    [Fact]
    public void TheQwenEntryStillAsksForTheQuantizedCache()
    {
        // Qwen 3.8 runs the same qwen35 graphs.
        List<CatalogModel> qwen = ModelCatalog.BuiltIn
            .Where(m => m.Family is CatalogFamily.Qwen35 or CatalogFamily.Qwen36 or CatalogFamily.Qwen38)
            .ToList();

        Assert.NotEmpty(qwen);
        foreach (CatalogModel m in qwen)
        {
            Assert.True(m.KvCacheDtype == "q8_0",
                $"{m.Id} is on {m.KvCacheDtype}: Qwen3.5 runs the fused graph, which reads a "
                + "block-quantized cache at decode parity for ~46% less KV memory");
        }
    }

    /// <summary>
    /// The drift guard, and the point of the whole file: a new text model added to the
    /// catalog without a context length would silently inherit the GGUF's, which is how
    /// this bug existed in the first place.
    /// </summary>
    [Fact]
    public void EveryModelThatHoldsAKvCacheStatesItsContextLength()
    {
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            if (m.Kind == CatalogArchitectureKind.Diffusion)
                continue;
            Assert.True(
                m.ContextLength > 0,
                $"{m.Id}: a text model must state ContextLength, or the engine falls back to the " +
                "GGUF's own window (262,144 for Qwen3.5) and the KV cache grows until jetsam");
            Assert.True(
                m.ContextLength <= 32768,
                $"{m.Id}: ContextLength {m.ContextLength} is beyond what a phone's jetsam budget holds");
        }
    }
}
