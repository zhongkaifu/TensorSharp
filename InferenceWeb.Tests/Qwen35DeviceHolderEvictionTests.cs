using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.5/3.8 per-request caches on one ggml_cuda device, against a simulated device: a holder's attention K/V is
/// "on the device" while it is DeviceMirrored, and demoting it frees that memory. On an RTX 5090 a Qwen3.8-27B
/// server with MAX_CONTEXT=32768 lost ~2.25 GB of free VRAM per request: every finished request retained its end
/// state with its device copies (a whole-window holder is 2 GiB there), and those copies were demoted only to make
/// room for a clone, so a few requests after a 16K prompt the fused graphs could not allocate and short_chat ran at
/// 4 tok/s. A new holder now makes room by demoting the least recently used retained holders first, a fresh holder
/// starts at 2048 rows as Flash Next's do, and a request that still does not fit waits.
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public class Qwen35DeviceHolderEvictionTests
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    /// <summary>A Qwen35Model whose device is simulated: free = Total - Base - the retained and live holders'
    /// mirrored K/V bytes.</summary>
    private sealed class DeviceModel : Qwen35Model
    {
        public DeviceModel() : base("", BackendType.Cpu) { }   // never called: GetUninitializedObject
        public long Total, Base;
        private List<Qwen35KvCacheHolder>? holders;   // GetUninitializedObject runs no field initializers
        public List<Qwen35KvCacheHolder> Holders => holders ??= new();
        public int Demotions;

        public long MirroredBytes()
        {
            long bytes = 0;
            foreach (var h in Holders)
                if (h.DeviceMirrored)
                    bytes += HolderBytes(h);
            return bytes;
        }

        public static long HolderBytes(Qwen35KvCacheHolder h)
        {
            long bytes = 0;
            foreach (Tensor[] set in new[] { h.K, h.V })
                foreach (Tensor? t in set)
                    if (t != null) bytes += t.Storage.ByteLength;
            return bytes;
        }

        public long Free => Total - Base - MirroredBytes();

        private protected override bool DeviceHolderCopiesEnabled => true;
        private protected override bool TryGetDeviceMemory(out long free, out long total)
        {
            total = Total;
            free = Free;
            return true;
        }
        private protected override bool CanCopyHolderKvOnDevice(Qwen35KvCacheHolder holder) => holder.DeviceMirrored;
        private protected override bool DemoteRetainedHolderKv(Qwen35KvCacheHolder holder)
        {
            holder.DeviceMirrored = false;
            Demotions++;
            return true;
        }
        private protected override long ReleaseKeptDeviceArenas() => 0;

        public Qwen35KvCacheHolder CreateFreshHolder()
        {
            var holder = (Qwen35KvCacheHolder)typeof(Qwen35Model).GetMethod("CreateFreshHolder", Instance)!.Invoke(this, null)!;
            Holders.Add(holder);
            return holder;
        }

        public void EnsureDeviceRoomForAnotherSequence(string requestId)
        {
            try { typeof(Qwen35Model).GetMethod("EnsureDeviceRoomForAnotherSequence", Instance)!.Invoke(this, new object[] { requestId }); }
            catch (TargetInvocationException ex) when (ex.InnerException != null) { throw ex.InnerException; }
        }
    }

    private sealed class Fixture : IDisposable
    {
        public DeviceModel Model { get; } = (DeviceModel)RuntimeHelpers.GetUninitializedObject(typeof(DeviceModel));
        public Dictionary<string, Qwen35Model.Qwen35KvCacheHolder> Retained { get; } = new(StringComparer.Ordinal);
        private readonly CpuAllocator allocator = new(BlasEnum.DotNet);
        private readonly EnvScope env = new();

        public Fixture(BackendType backend, string? kvInitialTokens, string reserveMb = "1")
        {
            env.Set("TS_KV_INITIAL_TOKENS", kvInitialTokens!);
            env.Set("TS_QWEN35_DEVICE_RETAIN_RESERVE_MB", reserveMb);
            Set(typeof(ModelBase), "<Config>k__BackingField", new ModelConfig { NumLayers = 2, NumKVHeads = 2, HiddenSize = 8, NumHeads = 2 });
            Set(typeof(ModelBase), "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(typeof(ModelBase), "_allocator", allocator);
            Set(typeof(ModelBase), "_backend", backend);
            Set(typeof(ModelBase), "_maxContextLength", 32768);
            Set(typeof(ModelBase), "_kvCacheDtype", KvCacheDtype.F32);
            Set(typeof(Qwen35Model), "_isRecurrent", new[] { false, false });
            Set(typeof(Qwen35Model), "_kvCacheK", new Tensor[2]);
            Set(typeof(Qwen35Model), "_initialKvCacheCapacity", 32768);
            Set(typeof(Qwen35Model), "_kvCacheCapacity", 32768);
            Set(typeof(Qwen35Model), "_convKernel", 2);
            foreach (string f in new[] { "_headKDim", "_numKHeads", "_headVDim", "_numVHeads" })
                Set(typeof(Qwen35Model), f, 1);
            Set(typeof(Qwen35Model), "_retainedFusedHolders", Retained);
            Set(typeof(Qwen35Model), "_fusedHolders", new Dictionary<string, Qwen35Model.Qwen35KvCacheHolder>(StringComparer.Ordinal));
        }

        private void Set(Type owner, string field, object value) => owner.GetField(field, Instance)!.SetValue(Model, value);

        public void Dispose()
        {
            foreach (var h in Model.Holders)
            {
                foreach (Tensor[] set in new[] { h.K, h.V })
                    foreach (Tensor? t in set) t?.Dispose();
                if (h.ConvScratch != IntPtr.Zero) System.Runtime.InteropServices.Marshal.FreeHGlobal(h.ConvScratch);
            }
            env.Dispose();
        }
    }

    [Fact]
    public void SequentialRequests_KeepTheRetainedDeviceCopiesWithinTheDevice()
    {
        // Whole-window holders (TS_KV_INITIAL_TOKENS = the window), as every holder was under MAX_CONTEXT before.
        using var fx = new Fixture(BackendType.GgmlCuda, kvInitialTokens: "32768");
        var model = fx.Model;
        model.Total = 64L << 20;
        model.Base = 32L << 20;
        long reserve = Qwen35Model.DeviceRetainReserveBytes(model.Total);
        Assert.Equal(1L << 20, reserve);

        long holderBytes = 0;
        for (int request = 0; request < 24; request++)
        {
            var holder = model.CreateFreshHolder();
            holderBytes = DeviceModel.HolderBytes(holder);
            holder.DeviceMirrored = true;              // its first step made its device copies
            fx.Retained["r" + request] = holder;       // it ended; its end state stays for prefix reuse
            Assert.True(model.Free >= reserve,
                $"request {request}: {model.Free >> 10} KiB free, below the {reserve >> 10} KiB the retained copies leave");
        }

        Assert.Equal(32768L * 2 * 4 * 4 * 2 * 2, holderBytes);   // rows x heads x dim x F32 x (K, V) x layers
        long fit = (model.Total - model.Base - reserve) / holderBytes;
        Assert.InRange(model.Holders.Count(h => h.DeviceMirrored), 1, (int)fit);
        Assert.Equal(24 - model.Holders.Count(h => h.DeviceMirrored), model.Demotions);
        // Least recently used first: the newest requests kept their copies.
        Assert.True(fx.Retained["r23"].DeviceMirrored);
        Assert.False(fx.Retained["r0"].DeviceMirrored);
    }

    [Theory]
    [InlineData(BackendType.GgmlCuda, null, 2048)]      // one CUDA device: start small, grow with the request
    [InlineData(BackendType.GgmlCuda, "32768", 32768)]  // an explicit TS_KV_INITIAL_TOKENS sizes holders as it sized the primary
    [InlineData(BackendType.GgmlCpu, null, 32768)]      // elsewhere: unchanged
    [InlineData(BackendType.GgmlMetal, null, 32768)]
    public void AFreshHolder_StartsAt2048RowsOnOneCudaDevice(BackendType backend, string? kvInitialTokens, int expected)
    {
        using var fx = new Fixture(backend, kvInitialTokens);
        fx.Model.Total = 64L << 30;

        var holder = fx.Model.CreateFreshHolder();

        Assert.Equal(expected, holder.KvCapacity);
    }

    [Fact]
    public void ARequestThatDoesNotFit_Waits_InsteadOfFailingMidStep()
    {
        using var fx = new Fixture(BackendType.GgmlCuda, kvInitialTokens: null);
        fx.Model.Total = 64L << 30;
        fx.Model.Base = fx.Model.Total - (512L << 20);   // 512 MiB free beside a 2 GiB headroom

        var ex = Assert.Throws<SequenceSlotUnavailableException>(() => fx.Model.EnsureDeviceRoomForAnotherSequence("late"));
        Assert.Contains("late waits for a running request to finish", ex.Message);

        fx.Model.Base = 16L << 30;
        fx.Model.EnsureDeviceRoomForAnotherSequence("late");   // room now: admitted
    }
}
