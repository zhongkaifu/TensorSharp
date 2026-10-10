using System.Text.RegularExpressions;
using TensorAgent.Core.Catalog;

namespace TensorAgent.Tests;

public sealed class CatalogTests
{
    [Fact]
    public void BuiltInContainsExactlyTheApprovedModelIds()
    {
        string[] expected =
        {
            "gemma-4-e2b-q8",
            "gemma-4-e4b-iq4xs",
            "gemma-4-12b-iq2m",
            "bonsai-2-27b-ptq1-0",
            "qwen3.5-9b-iq4xs",
            "qwen3.8-27b-q4kxl",
            "muse-glimmer-30b-q4kxl",
            "qwen3.8-flash-next-q2kxl",
            "qwen3.8-flash-next-iq1m",
            "qwen-image-2.1-q4km",
            "qwen-image-2.1-turbo-adq4k",
            "qwen-image-2.1-turbo-q8",
            "minimax-h3-fl2va-q4k",
            "minimax-h3-ref2va-q4k",
            "gemma-4-26b-a4b-qat-q4kxl",
            "gemma-4-31b-q4-0",
            "qwen3.5-35b-a3b-q4km",
            "qwen3.6-35b-a3b-q4km",
            "qwen3.6-27b-q4km",
            "gpt-oss-20b-mxfp4",
            "nemotron-h-8b-q4km",
            "nemotron-h-47b-q4km",
            "nemotron-3-nano-omni-30b-a3b-q4kxl",
            "nemotron-3.5-lightning-30b-a3b-mxfp4",
            "mistral-small-3.1-24b-q4km",
            "hy-mt2-1.8b-q4km",
            "deepseek-v4-flash-0731-q2kxl",
            "deepseek-v4.1-flash-engramq5-q2k",
            "glm-5.2-iq2xxs",
            "glm-5.3-q2kxl",
            "glm-5.3-flash-q2kxl",
            "diffusiongemma-26b-a4b-q4km",
            "wan2.1-t2v-1.3b-q8",
            "wan2.1-t2v-14b-q4km",
            "wan2.2-ti2v-5b-q8",
            "wan2.2-ti2v-5b-turbo-q8",
            "wan2.2-t2v-a14b-q4km",
            "wan2.2-i2v-a14b-q4km",
            "wan2.2-i2v-a14b-lightx2v-q4km",
        };

        Assert.Equal(expected, ModelCatalog.BuiltIn.Select(m => m.Id).ToArray());
    }

    [Fact]
    public void EveryEntryIsWellFormed()
    {
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            Assert.True(ids.Add(m.Id), $"duplicate id {m.Id}");
            Assert.Matches("^[a-z0-9.-]+$", m.Id);
            Assert.NotEmpty(m.DisplayName);
            Assert.NotEmpty(m.Files);
            Assert.Single(m.Files, f => f.Role == CatalogFileRole.Weights);
            Assert.False(m.Weights.Optional, $"{m.Id}: weights cannot be optional");
            foreach (CatalogFile f in m.Files)
            {
                if (m.SideloadOnly)
                {
                    Assert.Empty(f.Url);
                }
                else
                {
                    Assert.StartsWith("https://huggingface.co/", f.Url);
                    Assert.Matches(@"/resolve/(main|[0-9a-f]{40})/[^\s]+$", f.Url);
                }
                // A size read from a pointer file (~130 bytes) instead of the object is the
                // mistake this catches. Loose tokenizer files are genuinely small - MiniMax-H3's
                // tokenizer_config.json is 11 kB - so they are held to a kilobyte instead.
                Assert.True(f.Bytes > (f.Role == CatalogFileRole.Tokenizer ? 1_000 : 1_000_000),
                    $"{m.Id}/{f.FileName}: size {f.Bytes}");
                Assert.Matches("^[0-9a-f]{64}$", f.Sha256);
                Assert.False(f.FileName.Contains('/'), $"{m.Id}: file names are bare ({f.FileName})");
            }
            // Keep the recognized tiers narrow so a typo cannot silently expose an
            // entry on an unintended device class. 24, 32 and 48 are the desktop's: no
            // phone or tablet reaches them, so those entries stay off every one.
            Assert.Contains(m.MinDeviceMemoryGB, new[] { 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 });
            // A later shard of a split GGUF is required and carries shard 1's gguf-split
            // name with its own number: the engine finds it beside shard 1 by that name.
            var shards = m.Files.Where(f => f.Role == CatalogFileRole.WeightsShard).ToList();
            if (shards.Count > 0)
            {
                Match first = Regex.Match(m.Weights.FileName, @"^(?<prefix>.+)-00001-of-(?<count>\d{5})\.gguf$");
                Assert.True(first.Success, $"{m.Id}: shards need a -00001-of-NNNNN.gguf first file");
                Assert.Equal(int.Parse(first.Groups["count"].Value), shards.Count + 1);
                for (int i = 0; i < shards.Count; i++)
                {
                    Assert.False(shards[i].Optional, $"{m.Id}: shard {i + 2} is optional");
                    Assert.Equal($"{first.Groups["prefix"].Value}-{i + 2:D5}-of-{first.Groups["count"].Value}.gguf",
                        shards[i].FileName);
                }
            }
            Assert.NotEmpty(m.License);
        }
    }

    [Fact]
    public void ProjectorsAreDistinctFromWeights()
    {
        foreach (CatalogModel m in ModelCatalog.BuiltIn)
        {
            var names = m.Files.Select(f => f.FileName).ToList();
            Assert.Equal(names.Count, names.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        }
    }

    [Fact]
    public void Gemma4TwelveBUsesThePinnedIq2MArtifact()
    {
        CatalogModel model = Assert.Single(ModelCatalog.BuiltIn,
            m => m.Family == CatalogFamily.Gemma4 && m.Parameters == "12B");

        Assert.Equal("gemma-4-12b-iq2m", model.Id);
        Assert.Equal("UD-IQ2_M", model.Quantization);
        Assert.Equal("gemma-4-12b-it-UD-IQ2_M.gguf", model.Weights.FileName);
        Assert.Equal(4_213_353_280, model.Weights.Bytes);
        Assert.Equal("4bd2461d35398dbcf5f3d5f0c9ad91cac78ae35b556e3a81f315a0cc0815ae8c",
            model.Weights.Sha256);
    }

    /// <summary>
    /// What a device of a given size actually grants the app. A 12 GB iPhone gives
    /// about 8.5 GB with <c>com.apple.developer.kernel.increased-memory-limit</c>, and
    /// the fraction holds well enough across the range to gate a catalog with.
    /// </summary>
    private static double JetsamBudget(int deviceGB) => deviceGB * 1e9 * (8.5 / 12.0);

    /// <summary>
    /// The memory that is actually charged against the jetsam limit: ANONYMOUS memory
    /// only.
    ///
    /// <para>
    /// This used to be <c>Weights.Bytes + 0.5e9 + 2*projector</c>, on the stated premise
    /// that "Metal wires the mmap'd weights, so resident memory is roughly the GGUF".
    /// That premise is wrong, and measurably so. Weights are a file mapping, and Darwin
    /// charges mapped clean pages essentially nothing; ggml-metal wraps them with
    /// newBufferWithBytesNoCopy and the residency set does not fault them in. Measurements
    /// on this repo showed that treating the entire weights file as anonymous memory can
    /// overstate the charged footprint by roughly an order of magnitude.
    /// </para>
    ///
    /// <para>
    /// What IS anonymous: the KV cache (charged TWICE on Metal -- once for the host
    /// tensor and once for the Metal-side buffer, since the zero-copy wrap is refused for
    /// read-write tensors), the projector's dequantized copies (about twice its file),
    /// and the runtime plus graph scratch. 64 KiB/token is the per-token KV rate measured
    /// for Qwen3.5 9B and is used here as a conservative upper bound.
    /// </para>
    ///
    /// <para>
    /// The exception is a weights file the loader cannot map: Bonsai2's PTQ1_0 / PQ2_0
    /// packings are repacked losslessly to GGML Q2_0 at load, into anonymous memory that
    /// is about 29% / 6% larger than the file (<see cref="RepackedWeightBytes"/>).
    /// </para>
    /// </summary>
    private static double EstimatedAnonymous(CatalogModel model) =>
        0.5e9
        + model.ContextLength * 64.0 * 1024.0
        + (model.Projector is { Optional: false } p ? 2.0 * p.Bytes : 0)
        + RepackedWeightBytes(model);

    /// <summary>Anonymous memory the weights occupy when the loader repacks them instead
    /// of mapping the GGUF (zero for every mapped format).</summary>
    private static double RepackedWeightBytes(CatalogModel model) => model.Quantization switch
    {
        "PTQ1_0" => 1.29 * model.Weights.Bytes,
        "PQ2_0" => 1.06 * model.Weights.Bytes,
        _ => 0,
    };

    /// <summary>
    /// The other half, and the one the weights really answer to: they are not charged to
    /// jetsam, but they still have to be READABLE at a usable speed, which means the file
    /// has to fit the DEVICE's RAM alongside iOS rather than the app's jetsam budget.
    /// Past this line every token faults expert weights from flash. It is a performance
    /// bound, not a kill bound, which is why it is a separate number from
    /// <see cref="JetsamBudget"/> instead of folded into it.
    /// </summary>
    private static double WeightsResidencyCeiling(int deviceGB) => deviceGB * 1e9 * 0.87;

    [Fact]
    public void EveryTierStaysUnderTheJetsamBudgetOfTheSmallestDeviceItIsOfferedOn()
    {
        // Checking only the 12 GB tier let two entries through that could never load
        // on the tier they advertised: Gemma 4 E2B claimed 6 GB and needs 6.6, and
        // E4B Q4_K_XL claimed 8 and needs 6.8. A phone would have offered a five
        // gigabyte download and then been killed opening it — the worst outcome the
        // catalog can produce, because the user pays for it twice.
        foreach (CatalogModel m in ModelCatalog.BuiltIn.Where(m => !m.IsImageGenerator))
        {
            double anonymous = EstimatedAnonymous(m);
            double budget = JetsamBudget(m.MinDeviceMemoryGB);
            Assert.True(anonymous < budget,
                $"{m.Id} is offered at {m.MinDeviceMemoryGB} GB, which grants about "
                + $"{budget / 1e9:F1} GB, but charges about {anonymous / 1e9:F1} GB of anonymous memory");

            // Every shard counts (the first file alone is a split GGUF's metadata), less only
            // what the entry declares the engine pages from disk on demand.
            double ceiling = WeightsResidencyCeiling(m.MinDeviceMemoryGB);
            Assert.True(m.ResidentWeightsBytes < ceiling,
                $"{m.Id} is offered at {m.MinDeviceMemoryGB} GB but its resident weights are "
                + $"{m.ResidentWeightsBytes / 1e9:F1} GB, past the {ceiling / 1e9:F1} GB that device can "
                + "hold; it would fault every token from flash");
        }
    }

    [Fact]
    public void EveryModelADeviceIsOfferedFitsThatDevice()
    {
        // The other direction: whatever ForDevice hands back must fit the device that
        // asked, for every size a real iPhone comes in.
        foreach (int deviceGB in new[] { 6, 8, 12, 16 })
        {
            foreach (CatalogModel m in ModelCatalog.ForDevice(deviceGB).Where(m => !m.IsImageGenerator))
            {
                Assert.True(EstimatedAnonymous(m) < JetsamBudget(deviceGB),
                    $"a {deviceGB} GB device is offered {m.Id}, which charges about "
                    + $"{EstimatedAnonymous(m) / 1e9:F1} GB against a {JetsamBudget(deviceGB) / 1e9:F1} GB budget");
                Assert.True(m.ResidentWeightsBytes < WeightsResidencyCeiling(deviceGB),
                    $"a {deviceGB} GB device is offered {m.Id}, whose {m.ResidentWeightsBytes / 1e9:F1} GB of "
                    + $"weights exceed the {WeightsResidencyCeiling(deviceGB) / 1e9:F1} GB it can hold");
            }
        }
    }

    private static readonly string[] DesktopOnly =
    {
        "qwen3.8-27b-q4kxl",
        "muse-glimmer-30b-q4kxl",
        "qwen3.8-flash-next-q2kxl",
        "qwen3.8-flash-next-iq1m",
        "qwen-image-2.1-q4km",
        "qwen-image-2.1-turbo-adq4k",
        "qwen-image-2.1-turbo-q8",
        "minimax-h3-fl2va-q4k",
        "minimax-h3-ref2va-q4k",
    };

    [Fact]
    public void DeviceTiersHideTheCatalogBelowTwelveGbAndHoldBonsai2ForSixteenGb()
    {
        Assert.Empty(ModelCatalog.ForDevice(8));
        // Bonsai 2 27B is the one 16 GB entry: its repacked weights do not fit a 12 GB phone.
        Assert.Equal(
            ModelCatalog.BuiltIn.Where(m => m.MinDeviceMemoryGB <= 12).Select(m => m.Id),
            ModelCatalog.ForDevice(12).Select(m => m.Id));
        Assert.Equal(
            ModelCatalog.BuiltIn.Where(m => m.MinDeviceMemoryGB <= 16).Select(m => m.Id),
            ModelCatalog.ForDevice(16).Select(m => m.Id));
    }

    /// <summary>
    /// Qwen3.8 27B, Muse-Glimmer 30B and Qwen-Image 2.1 are offered only where a Mac's
    /// memory exists (no iPhone or iPad reaches 24 GB): the two chat models from 32 GB,
    /// the image model and its 4-bit Turbo from 24, the 8-bit Turbo from 32. See <see cref="EachDesktopEntryFitsItsTierBesideMacOS"/>
    /// for the measurements behind each number.
    /// </summary>
    [Theory]
    [InlineData("qwen3.8-27b-q4kxl", 32)]
    [InlineData("muse-glimmer-30b-q4kxl", 32)]
    [InlineData("qwen-image-2.1-q4km", 24)]
    [InlineData("qwen-image-2.1-turbo-adq4k", 24)]
    [InlineData("qwen-image-2.1-turbo-q8", 32)]
    [InlineData("minimax-h3-fl2va-q4k", 32)]
    [InlineData("minimax-h3-ref2va-q4k", 32)]
    [InlineData("qwen3.8-flash-next-q2kxl", 48)]
    [InlineData("qwen3.8-flash-next-iq1m", 32)]
    public void TheDesktopTiersHoldTheModelsNoPhoneCanRunWell(string id, int tier)
    {
        CatalogModel model = Assert.IsType<CatalogModel>(ModelCatalog.Find(id));
        Assert.Equal(tier, model.MinDeviceMemoryGB);
        Assert.Contains(id, DesktopOnly);
        Assert.DoesNotContain(ModelCatalog.ForDevice(16), m => m.Id == id);
        Assert.Contains(ModelCatalog.ForDevice(tier), m => m.Id == id);
        Assert.Contains(ModelCatalog.ForDevice(48), m => m.Id == id);
        if (tier > 24)
            Assert.DoesNotContain(ModelCatalog.ForDevice(24), m => m.Id == id);
        if (tier > 32)
            Assert.DoesNotContain(ModelCatalog.ForDevice(32), m => m.Id == id);
    }

    /// <summary>
    /// Paging weights from disk is a property of one architecture's engine, not a way to
    /// fit any big file: only a mixture of experts on a desktop tier may declare it, and
    /// never for the whole file.
    /// </summary>
    [Fact]
    public void OnlyADesktopMixtureOfExpertsDeclaresWeightsPagedFromDisk()
    {
        var approved = new Dictionary<string, int>
        {
            ["qwen3.8-flash-next-q2kxl"] = 48,
            ["qwen3.8-flash-next-iq1m"] = 32,
        };
        foreach (CatalogModel m in ModelCatalog.BuiltIn.Where(m => m.WeightsPagedFromDiskBytes != 0))
        {
            Assert.Equal(CatalogArchitectureKind.MixtureOfExperts, m.Kind);
            Assert.True(approved.TryGetValue(m.Id, out int tier), $"{m.Id} has no approved paging tier");
            Assert.Equal(tier, m.MinDeviceMemoryGB);
            Assert.InRange(m.WeightsPagedFromDiskBytes, 1, m.WeightsBytes - 1);
        }
        Assert.Equal(approved.Keys.OrderBy(id => id),
            ModelCatalog.BuiltIn.Where(m => m.WeightsPagedFromDiskBytes != 0).Select(m => m.Id).OrderBy(id => id));
    }

    /// <summary>The residency checks read every shard: a split file that declares nothing paged
    /// is held to its whole size, not to its 11 MB first file.</summary>
    [Fact]
    public void ASplitEntryIsHeldToAllItsShards()
    {
        CatalogModel paged = ModelCatalog.Find("qwen3.8-flash-next-q2kxl")!;
        CatalogModel undeclared = paged with { WeightsPagedFromDiskBytes = 0 };
        Assert.Equal(78_869_128_864, undeclared.ResidentWeightsBytes);
        Assert.True(undeclared.ResidentWeightsBytes > WeightsResidencyCeiling(undeclared.MinDeviceMemoryGB));
        Assert.True(paged.ResidentWeightsBytes < WeightsResidencyCeiling(paged.MinDeviceMemoryGB));
    }

    [Fact]
    public void ADesktopIsOfferedEverythingASmallerDeviceIs()
    {
        foreach (int tier in new[] { 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024 })
            Assert.All(ModelCatalog.ForDevice(tier), model => Assert.Contains(model, ModelCatalog.ForDevice(1024)));
        Assert.Equal(ModelCatalog.BuiltIn.Select(m => m.Id), ModelCatalog.ForDevice(1024).Select(m => m.Id));
    }

    /// <summary>
    /// What each desktop entry needs, MEASURED in the Mac app on 2026-09-30 (Apple M5 Pro,
    /// 48 GB, ggml_metal): the mapped files the model reads while it works, and the app's
    /// footprint at its highest. A Mac has no jetsam to kill the app, so nothing else in
    /// this file checks a desktop entry; but a dense model reads every weight for every
    /// token, and once the weights cannot stay resident beside the app and macOS, every
    /// token pages them back in from disk.
    /// </summary>
    private static readonly Dictionary<string, (double ResidentFilesGB, double FootprintGB, string How)> MeasuredOnAMac = new()
    {
        ["qwen3.8-27b-q4kxl"] = (17.56, 8.8, "chat-e2e.py's seven scenarios with the projector, LeanCaches"),
        ["muse-glimmer-30b-q4kxl"] = (15.88, 10.9, "chat-e2e.py's seven scenarios with the projector, LeanCaches"),
        // The DiT stays mapped through the denoise; the text encoder is released first.
        ["qwen-image-2.1-q4km"] = (4.19, 14.3, "an edit at 1248x832, 40 steps (the CLI's peak footprint)"),
        // Measured 2026-10-09 the same way (/usr/bin/time -l, peak memory footprint), 8 steps: the
        // peak does not depend on the transformer's quantization (base 14.37, AD-Q4_K 14.35,
        // Q8_0 14.36 GB), so the 8-bit file's 3.4 GB of extra weights are what move it up a tier.
        ["qwen-image-2.1-turbo-adq4k"] = (4.20, 14.4, "an edit at 1248x832, 8 steps (the CLI's peak footprint)"),
        ["qwen-image-2.1-turbo-q8"] = (7.59, 14.4, "an edit at 1248x832, 8 steps (the CLI's peak footprint)"),
        // The largest stage is the 18.2 GB text encoder: the denoiser and the VAEs kept from the
        // previous clip are taken off the device before it runs (MiniMaxH3Pipeline), and wired
        // memory peaked at 20 GB with the decode (denoiser + video VAE) on top of the system's.
        // A photo (keyframe or reference) is what takes the footprint to its highest: the
        // video VAE's encoder converts its kernels to F32 in managed memory.
        ["minimax-h3-fl2va-q4k"] = (18.22, 2.24, "chat-e2e.py's film and animate in the Mac app, 22 frames, 20 steps"),
        ["minimax-h3-ref2va-q4k"] = (18.22, 2.23, "chat-e2e.py's reference in the Mac app, 22 frames, 20 steps"),
        // Measured 2026-10-01 in the Debug Mac app: the files are what stays resident (the dense
        // half and the 15 layers' experts the engine keeps on the GPU); the rest is read from the
        // SSD (CatalogModel.WeightsPagedFromDiskBytes). The footprint is phys_footprint_peak over
        // the warm-up of the 7.2k-token agent prompt and chat-e2e.py's six text scenarios.
        ["qwen3.8-flash-next-q2kxl"] = (18.34, 7.31, "chat-e2e.py's text scenarios in the Mac app, phys_footprint_peak"),
    };

    // IQ1_M's 32 GB tier was validated on Windows with 16 GB CUDA VRAM and SSD paging.
    // It has no measured Mac footprint; do not present the Windows evidence as Mac data.
    private static readonly string[] ValidatedOnWindowsOnly = { "qwen3.8-flash-next-iq1m" };

    [Fact]
    public void DesktopValidationScopesCoverEachEntryWithoutInventingMacMeasurements()
    {
        Assert.Empty(MeasuredOnAMac.Keys.Intersect(ValidatedOnWindowsOnly));
        Assert.Equal(DesktopOnly.OrderBy(id => id),
            MeasuredOnAMac.Keys.Concat(ValidatedOnWindowsOnly).OrderBy(id => id));
        Assert.All(ValidatedOnWindowsOnly, id => Assert.True(ModelCatalog.Find(id)!.Experimental));
    }

    /// <summary>What macOS and the rest of a desktop keep for themselves.</summary>
    private const double MacOsGB = 5.0;

    [Fact]
    public void EachDesktopEntryFitsItsTierBesideMacOS()
    {
        Assert.Equal(DesktopOnly.Except(ValidatedOnWindowsOnly).OrderBy(id => id), MeasuredOnAMac.Keys.OrderBy(id => id));
        int[] tiers = { 6, 8, 12, 16, 24, 32, 48 };
        foreach ((string id, (double files, double footprint, string how)) in MeasuredOnAMac)
        {
            CatalogModel model = ModelCatalog.Find(id)!;
            double need = files + footprint + MacOsGB;
            if (model.WeightsPagedFromDiskBytes > 0)
            {
                // An entry that pages weights from disk is held to its resident part, and what is
                // left must still cache a third of the experts it pages (a token reads few rows
                // of the n-gram table, so that part is left out). The next tier down is not
                // checked the usual way: a smaller Mac would have to page nearly everything.
                Assert.Equal(model.ResidentWeightsBytes / 1e9, files, 2);
                Assert.True(need <= model.MinDeviceMemoryGB,
                    $"{id} needs about {need:F1} GB resident ({how}) but is offered from {model.MinDeviceMemoryGB} GB");
                double pagedExperts = (model.WeightsPagedFromDiskBytes - 28_800_138_240) / 1e9;
                Assert.True(model.MinDeviceMemoryGB - need >= pagedExperts / 3,
                    $"{id} leaves {model.MinDeviceMemoryGB - need:F1} GB of page cache for {pagedExperts:F1} GB of paged experts");
                Assert.Equal(tiers.Max(), model.MinDeviceMemoryGB);
                continue;
            }
            Assert.True(need <= model.MinDeviceMemoryGB,
                $"{id} needs about {need:F1} GB ({how}) but is offered from {model.MinDeviceMemoryGB} GB");
            // And not offered higher than it needs: the next tier down must really be too small.
            int below = tiers.Where(t => t < model.MinDeviceMemoryGB).DefaultIfEmpty(0).Max();
            Assert.True(need > below,
                $"{id} needs about {need:F1} GB ({how}), which the {below} GB tier already holds");
        }
    }

    /// <summary>
    /// Every chat entry has room on a desktop for TensorAgent's shared prompt (~7.2k tokens
    /// of tools, skills and instructions), a reply and a conversation. Eighteen entries said
    /// 8,192 -- a phone's budget -- and on a Mac that left about a thousand tokens beside the
    /// shared prompt, so every follow-up compacted the conversation away. The window is the
    /// documented rule (<see cref="CatalogModel.DesktopContextLength"/>), checked here
    /// against each entry's stated K/V cost; the phone's <see cref="CatalogModel.ContextLength"/>
    /// and the jetsam tests above are unchanged.
    /// </summary>
    [Fact]
    public void EveryChatEntryHasRoomForTheSharedPromptAndAConversationOnTheDesktop()
    {
        foreach (CatalogModel m in ModelCatalog.BuiltIn.Where(m => m.Kind != CatalogArchitectureKind.Diffusion))
        {
            Assert.True(m.DesktopContextLength >= CatalogModel.MinimumDesktopChatContext,
                $"{m.Id} gets a {m.DesktopContextLength}-token window on a desktop; the shared prompt, a reply "
                + $"and a conversation need {CatalogModel.MinimumDesktopChatContext}");
            Assert.True(m.DesktopContextLength >= m.ContextLength, $"{m.Id}: a desktop never gets less than a phone");
            if (m.ContextLength >= CatalogModel.DesktopChatContextTarget)
                continue;

            Assert.True(m.KvBytesPerToken > 0, $"{m.Id} must state its K/V bytes per token to be given a desktop window");
            Assert.True(m.DesktopContextLength <= CatalogModel.DesktopChatContextTarget);
            Assert.Equal(0, m.DesktopContextLength % 4096);
            // Half of what the tier has beside the weights, the dequantized projector and the
            // system holds the whole window's K/V twice (host tensor and device mirror).
            double spare = m.MinDeviceMemoryGB * 1e9 - m.ResidentWeightsBytes
                - 2.0 * (m.Projector?.Bytes ?? 0) - MacOsGB * 1e9;
            Assert.True(2.0 * m.KvBytesPerToken * m.DesktopContextLength <= spare / 2,
                $"{m.Id}: {m.DesktopContextLength} tokens of K/V do not fit its {m.MinDeviceMemoryGB} GB tier");
        }

        // Where the tier, not the target, decides -- each worked in the entry's comment.
        Assert.Equal(16384, ModelCatalog.Find("gemma-4-e4b-iq4xs")!.DesktopContextLength);
        Assert.Equal(20480, ModelCatalog.Find("gemma-4-31b-q4-0")!.DesktopContextLength);
        Assert.Equal(28672, ModelCatalog.Find("qwen3.6-27b-q4km")!.DesktopContextLength);
        Assert.Equal(16384, ModelCatalog.Find("mistral-small-3.1-24b-q4km")!.DesktopContextLength);
        Assert.Equal(32768, ModelCatalog.Find("gemma-4-e2b-q8")!.DesktopContextLength);
        Assert.Equal(32768, ModelCatalog.Find("nemotron-h-8b-q4km")!.DesktopContextLength);
        // A diffusion entry's prompt carries no shared agent prompt, and its window is its own.
        CatalogModel diffusion = ModelCatalog.Find("diffusiongemma-26b-a4b-q4km")!;
        Assert.Equal(diffusion.ContextLength, diffusion.DesktopContextLength);
    }

    [Fact]
    public void TheNewDesktopEntriesUseThePinnedFourBitArtifacts()
    {
        CatalogModel qwen = ModelCatalog.Find("qwen3.8-27b-q4kxl")!;
        Assert.Equal(CatalogFamily.Qwen38, qwen.Family);
        Assert.Equal("Qwen3.8-27B-UD-Q4_K_XL.gguf", qwen.Weights.FileName);
        Assert.Equal(17_559_178_144, qwen.Weights.Bytes);
        Assert.Equal("3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e", qwen.Weights.Sha256);
        Assert.True(qwen.Projector is { Optional: true });

        CatalogModel muse = ModelCatalog.Find("muse-glimmer-30b-q4kxl")!;
        Assert.Equal(CatalogFamily.MuseGlimmer, muse.Family);
        Assert.Equal("Muse-Glimmer-30B-UD-Q4_K_XL.gguf", muse.Weights.FileName);
        Assert.Equal(15_878_222_368, muse.Weights.Bytes);
        Assert.Equal("82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38", muse.Weights.Sha256);
        Assert.True(muse.Projector is { Optional: true });
        Assert.True(Assert.Single(muse.Files, f => f.Role == CatalogFileRole.Draft).Optional);
        Assert.True(Assert.Single(qwen.Files, f => f.Role == CatalogFileRole.Draft).Optional);
    }

    [Theory]
    [InlineData(11_560_000_000L, 12)]
    [InlineData(12_000_000_000L, 12)]
    [InlineData(8_100_000_000L, 8)]
    [InlineData(7_700_000_000L, 8)]
    [InlineData(16_700_000_000L, 16)]
    [InlineData(5_800_000_000L, 6)]
    public void MemoryTierRoundsToTheMarketingNumber(long bytes, int tier)
    {
        Assert.Equal(tier, ModelCatalog.DeviceMemoryTier(bytes));
    }

    /// <summary>
    /// A weights file sitting loose in the models directory is swept too.
    ///
    /// <para>
    /// The directory holds one sub-folder per catalog id and nothing else, so a file
    /// directly inside it belongs to no entry by construction. It gets there when
    /// weights are pushed onto the device by hand and land beside the per-model folders
    /// instead of inside one — and it is worse off than an orphaned directory, because
    /// the Models list is built from catalog entries: a stray file has no row, no size
    /// attributed to any model, and no delete button, while being several gigabytes.
    /// </para>
    /// <para>
    /// The other half is that a real model's files are NOT strays. They live one level
    /// down, so enumerating only the top level must not reach them.
    /// </para>
    /// </summary>
    [Fact]
    public void ASweepAlsoRemovesAWeightsFileLeftLooseInTheModelsDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "ta-stray-" + Guid.NewGuid().ToString("n"));
        try
        {
            var store = new ModelStore(root);
            CatalogModel offered = ModelCatalog.ForDevice(12)[0];

            Directory.CreateDirectory(Path.Combine(root, offered.Id));
            File.WriteAllBytes(Path.Combine(root, offered.Id, "weights.gguf"), new byte[4096]);
            File.WriteAllBytes(Path.Combine(root, "gemma-4-12b-it-UD-IQ2_M.gguf"), new byte[1024]);
            File.WriteAllBytes(Path.Combine(root, "mmproj-F16.gguf"), new byte[512]);

            long freed = store.SweepOrphanedModels();

            Assert.Equal(1536, freed);
            Assert.False(File.Exists(Path.Combine(root, "gemma-4-12b-it-UD-IQ2_M.gguf")));
            Assert.False(File.Exists(Path.Combine(root, "mmproj-F16.gguf")));
            Assert.True(File.Exists(Path.Combine(root, offered.Id, "weights.gguf")),
                "the sweep reached inside a model's own directory and deleted its weights");
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>
    /// Weights left behind when an entry changes which file it points at.
    ///
    /// <para>
    /// The id carries the quantization, so re-pointing Gemma 4 12B from UD-IQ3_XXS to
    /// UD-IQ2_M renames its directory and orphans the old one -- 4.6 GB with no row in
    /// the Models list and therefore no way for the user to remove it. An entry gated
    /// to a bigger device is NOT an orphan, which is the half that would be a data-loss
    /// bug: an iPad-only model's weights must survive a sweep run on a phone.
    /// </para>
    /// </summary>
    [Fact]
    public void ASweepRemovesWeightsNoEntryClaimsAndKeepsTheOnesThatAreMerelyGatedOff()
    {
        string root = Path.Combine(Path.GetTempPath(), "ta-sweep-" + Guid.NewGuid().ToString("n"));
        try
        {
            var store = new ModelStore(root);
            CatalogModel offered = ModelCatalog.ForDevice(12)[0];
            CatalogModel gatedOff = offered with
            {
                Id = "synthetic-16gb-entry",
                MinDeviceMemoryGB = 16,
            };
            CatalogModel[] wholeCatalog = { offered, gatedOff };

            foreach (string id in new[] { offered.Id, gatedOff.Id, "gemma-4-12b-iq3xxs" })
            {
                Directory.CreateDirectory(Path.Combine(root, id));
                File.WriteAllBytes(Path.Combine(root, id, "weights.gguf"), new byte[2048]);
            }

            long freed = store.SweepOrphanedModels(wholeCatalog);

            Assert.Equal(2048, freed);
            Assert.True(Directory.Exists(Path.Combine(root, offered.Id)));
            Assert.True(Directory.Exists(Path.Combine(root, gatedOff.Id)),
                "a sweep on a phone deleted the weights of a model only an iPad is offered");
            Assert.False(Directory.Exists(Path.Combine(root, "gemma-4-12b-iq3xxs")));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    /// <summary>
    /// What a build OLDER than the catalog that installed a model finds: a folder its own
    /// catalog has never listed.
    ///
    /// <para>
    /// On a Mac the Debug and Release builds share one models directory. A Release build
    /// from 2026-09-30, launched after the Debug build had installed the five desktop
    /// entries added later that day, treated every id it did not know as retired and
    /// deleted all five. An unknown id is kept now; only a retired one is reclaimed.
    /// </para>
    /// </summary>
    [Fact]
    public void ASweepByAnOlderBuildKeepsTheModelsANewerBuildInstalled()
    {
        string root = Path.Combine(Path.GetTempPath(), "ta-skew-" + Guid.NewGuid().ToString("n"));
        try
        {
            // The catalog as it was before the desktop entries existed.
            CatalogModel[] older = ModelCatalog.BuiltIn.Where(m => !DesktopOnly.Contains(m.Id)).ToArray();
            var store = new ModelStore(root, catalog: older);
            foreach (string id in DesktopOnly.Append("gemma-4-12b-iq3xxs"))
            {
                Directory.CreateDirectory(Path.Combine(root, id));
                File.WriteAllBytes(Path.Combine(root, id, "weights.gguf"), new byte[1024]);
            }

            long freed = store.SweepOrphanedModels();

            Assert.Equal(1024, freed);
            foreach (string id in DesktopOnly)
                Assert.True(Directory.Exists(Path.Combine(root, id)), $"a build older than {id} deleted its weights");
            Assert.False(Directory.Exists(Path.Combine(root, "gemma-4-12b-iq3xxs")));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch { }
        }
    }

    /// <summary>
    /// Every id the catalog has ever shipped, built-in or retired. It only grows: a new
    /// entry's id is added here, and an id that leaves <see cref="ModelCatalog.BuiltIn"/>
    /// stays here and must move to <see cref="ModelCatalog.Retired"/>.
    /// </summary>
    private static readonly string[] Shipped =
    {
        "gemma-4-e2b-q8",
        "gemma-4-e4b-q8",
        "gemma-4-e4b-q4kxl",
        "gemma-4-e4b-iq4xs",
        "gemma-4-12b-q4kxl",
        "gemma-4-12b-iq3xxs",
        "gemma-4-12b-iq2m",
        "gemma-4-26b-a4b-iq2xxs",
        "gpt-oss-20b-q8",
        "bonsai-8b-q1-0",
        "bonsai-27b-q1-0",
        "bonsai-2-27b-ptq1-0",
        "qwen3.5-9b-q4kxl",
        "qwen3.5-9b-iq4xs",
        "qwen3.6-35b-a3b-iq1m",
        "qwen3.8-27b-iq2xxs",
        "qwen3.8-27b-iq1s",
        "qwen3.8-27b-q4kxl",
        "muse-glimmer-30b-q4kxl",
        "qwen-image-edit-2511-q2k",
        "qwen-image-2.1-q4km",
        "minimax-h3-fl2va-q4k",
        "minimax-h3-ref2va-q4k",
        "qwen3.8-flash-next-q2kxl",
        "qwen3.8-flash-next-iq1m",
        "qwen-image-2.1-turbo-adq4k",
        "qwen-image-2.1-turbo-q8",
        "gemma-4-26b-a4b-qat-q4kxl",
        "gemma-4-31b-q4-0",
        "qwen3.5-35b-a3b-q4km",
        "qwen3.6-35b-a3b-q4km",
        "qwen3.6-27b-q4km",
        "gpt-oss-20b-mxfp4",
        "nemotron-h-8b-q4km",
        "nemotron-h-47b-q4km",
        "nemotron-3-nano-omni-30b-a3b-q4kxl",
        "nemotron-3.5-lightning-30b-a3b-mxfp4",
        "mistral-small-3.1-24b-q4km",
        "hy-mt2-1.8b-q4km",
        "deepseek-v4-flash-0731-q2kxl",
        "deepseek-v4.1-flash-engramq5-q2k",
        "glm-5.2-iq2xxs",
        "glm-5.3-q2kxl",
        "glm-5.3-flash-q2kxl",
        "diffusiongemma-26b-a4b-q4km",
        "wan2.1-t2v-1.3b-q8",
        "wan2.1-t2v-14b-q4km",
        "wan2.2-ti2v-5b-q8",
        "wan2.2-ti2v-5b-turbo-q8",
        "wan2.2-t2v-a14b-q4km",
        "wan2.2-i2v-a14b-q4km",
        "wan2.2-i2v-a14b-lightx2v-q4km",
    };

    /// <summary>
    /// The launch sweep reclaims only <see cref="ModelCatalog.Retired"/> ids
    /// (<see cref="ASweepByAnOlderBuildKeepsTheModelsANewerBuildInstalled"/>), so an entry
    /// that simply vanished from the catalog would leave its gigabytes on every device
    /// that installed it, with no row in the Models list to delete them from. Both
    /// directions are checked: an id that left without being retired, and an entry added
    /// without being recorded here.
    /// </summary>
    [Fact]
    public void EveryIdTheCatalogHasShippedIsBuiltInOrRetired()
    {
        string[] current = ModelCatalog.BuiltIn.Select(m => m.Id).ToArray();
        Assert.Empty(current.Intersect(ModelCatalog.Retired, StringComparer.OrdinalIgnoreCase));
        Assert.Equal(ModelCatalog.Retired.Count, ModelCatalog.Retired.Distinct(StringComparer.OrdinalIgnoreCase).Count());
        foreach (string id in ModelCatalog.Retired)
            Assert.Matches("^[a-z0-9.-]+$", id);

        var listed = new HashSet<string>(current.Concat(ModelCatalog.Retired), StringComparer.OrdinalIgnoreCase);
        string[] vanished = Shipped.Where(id => !listed.Contains(id)).ToArray();
        Assert.True(vanished.Length == 0,
            $"{string.Join(", ", vanished)} left ModelCatalog.BuiltIn without moving to ModelCatalog.Retired; "
            + "no launch would ever reclaim the weights installed under that name");
        string[] unrecorded = listed.Where(id => !Shipped.Contains(id, StringComparer.OrdinalIgnoreCase)).ToArray();
        Assert.True(unrecorded.Length == 0,
            $"{string.Join(", ", unrecorded)} is not in CatalogTests.Shipped; add every new entry's id there");
        Assert.True(ModelCatalog.IsRetired("GEMMA-4-12B-IQ3XXS"));
        Assert.False(ModelCatalog.IsRetired("gemma-4-12b-iq2m"));
    }

    [Fact]
    public void FindIsCaseInsensitive()
    {
        Assert.NotNull(ModelCatalog.Find("GEMMA-4-E4B-IQ4XS"));
        Assert.Null(ModelCatalog.Find("nope"));
    }
}
