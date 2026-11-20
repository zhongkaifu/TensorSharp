using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// The Liquid LFM3 entries: the files config/lfm3-1.7b.json and config/lfm3-omni-2b.json pin,
/// the phone tier, the K/V the runtime allocates, and the card sampling the app sends.
/// </summary>
public sealed class Lfm3CatalogTests
{
    private const string TextRepo = "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/";
    private const string OmniRepo = "https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF/resolve/d508b88f08a5b4286f2d2805a6a8d69a0ff44c72/";

    [Fact]
    public void TheTextEntryPinsTheConfigsQ4KmFile()
    {
        CatalogModel model = ModelCatalog.Find("lfm3-1.7b-q4km")!;
        Assert.Equal(CatalogFamily.Lfm3, model.Family);
        Assert.Equal("LFM3 1.7B", model.DisplayName);
        CatalogFile weights = Assert.Single(model.Files);
        Assert.Equal(TextRepo + "LFM3-1.7B-Q4_K_M.gguf", weights.Url);
        Assert.Equal(1_099_627_808, weights.Bytes);
        Assert.Equal("3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4", weights.Sha256);
        Assert.Equal(CatalogModalities.Text, model.Modalities);
        Assert.True(model.SupportsThinking);
        // The card's general preset; the penalty is the engine's 1.1 default.
        Assert.Equal(new CatalogSampling(0.1f, 50, 1.0f, 0.0f), model.Sampling);
        Assert.Null(model.Sampling.RepetitionPenalty);
        Assert.Equal(1.1f, AgentAppHost.SamplingDefaultsFor(model).Values.RepetitionPenalty);
    }

    [Fact]
    public void TheOmniEntryTakesImagesVideoAndAudioInAndNoSoundOut()
    {
        CatalogModel model = ModelCatalog.Find("lfm3-omni-2b-q4km")!;
        Assert.Equal(CatalogFamily.Lfm3, model.Family);
        Assert.Equal(OmniRepo + "LFM3-Omni-2B-Q4_K_M.gguf", model.Weights.Url);
        Assert.Equal(1_099_630_080, model.Weights.Bytes);
        Assert.Equal("9f8d30e9d4fd7a9349d586e077988229b3ca0a0e4ad730897563d1eccc9893e2", model.Weights.Sha256);
        CatalogFile projector = Assert.IsType<CatalogFile>(model.Projector);
        Assert.False(projector.Optional);
        Assert.Equal(OmniRepo + "mmproj-LFM3-Omni-2B-Q8_0.gguf", projector.Url);
        Assert.Equal(506_842_528, projector.Bytes);
        Assert.Equal("8a25f9d1931244429de404fccacfb7fb933970a36aa5688d2f5069f29c6b27b1", projector.Sha256);
        Assert.Equal(CatalogModalities.Image | CatalogModalities.Video | CatalogModalities.Audio, model.Modalities);
        // AudioOutput would list it as a video generator with sound and drop "chat".
        Assert.False(model.Modalities.HasFlag(CatalogModalities.AudioOutput));
        Assert.False(model.IsVideoGenerator);
        Assert.False(model.SupportsThinking);
        // The LFM3 card's general preset, as the text entry: greedy decoding without a penalty
        // (the Omni card's) made replies repeat a sentence over and over (eng/lfm3/loop).
        Assert.Equal(new CatalogSampling(0.1f, 50, 1.0f, 0.0f), model.Sampling);
        var defaults = AgentAppHost.SamplingDefaultsFor(model).Values;
        Assert.Equal(0.1f, defaults.Temperature);
        Assert.Equal(50, defaults.TopK);
        Assert.Equal(1.1f, defaults.RepetitionPenalty);
    }

    [Theory]
    [InlineData("lfm3-1.7b-q4km")]
    [InlineData("lfm3-omni-2b-q4km")]
    public void BothEntriesAreExperimentalPhoneTierModelsWithTheRuntimesKvCost(string id)
    {
        CatalogModel model = ModelCatalog.Find(id)!;
        Assert.Equal(CatalogArchitectureKind.Dense, model.Kind);
        Assert.Equal(12, model.MinDeviceMemoryGB);
        Assert.Contains(ModelCatalog.ForDevice(12), m => m.Id == id);
        // The six full-attention layers (four KV heads of 128, F16); the five sliding-window layers keep fixed rings.
        Assert.Equal(6 * 4 * 128 * 2 * 2, model.KvBytesPerToken);
        Assert.Equal(32768, model.ContextLength);
        Assert.Equal(32768, model.DesktopContextLength);
        Assert.Equal("f16", model.KvCacheDtype);
        Assert.True(model.Experimental);
        Assert.Equal("LFM Open License v1.0", model.License);
        Assert.Equal("lfm", ModelBrowser.FamilyId(model));
    }
}
