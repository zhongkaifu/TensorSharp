using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.5/3.8 on the BatchedPaged plan: when a second request joins a request that has been running alone (the N=1
/// linear path), the executor migrates the first one's linear K/V and recurrent state into paged storage
/// (TryMigrateLinearKVToPaged), reading the host mirrors. A prompt prefill chunk or a speculative verify through the
/// fused verify graph leaves the live recurrent state on the device (_fvDeviceStateCurrent) with those mirrors behind
/// and no fused decode resident, and the migration's drain skipped that case. On ggml_cuda with TS_PER_SEQ_FUSED=0 the
/// request that arrived a step ahead of its parallel_4 peers then answered "PRODUCTS\n# 2025年1月1日 新年のご挨拶"
/// from its second token: it decoded from its pre-prompt recurrent state.
/// </summary>
public class Qwen35MigrationDrainTests
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;

    /// <summary>Records drains; the device state is simulated by the flag alone.</summary>
    private sealed class DrainModel : Qwen35Model
    {
        public DrainModel() : base("", BackendType.Cpu) { }   // never called: GetUninitializedObject
        public int Drains;
        internal override void DrainDeviceRecurrentState()
        {
            if (!(bool)typeof(Qwen35Model).GetField("_fvDeviceStateCurrent", Instance)!.GetValue(this)!)
                return;
            Drains++;
            typeof(Qwen35Model).GetField("_fvDeviceStateCurrent", Instance)!.SetValue(this, false);
        }
    }

    private static DrainModel ModelWithVerifyCurrentState(bool deviceStateCurrent)
    {
        var model = (DrainModel)RuntimeHelpers.GetUninitializedObject(typeof(DrainModel));
        typeof(ModelBase).GetField("_backend", Instance)!.SetValue(model, BackendType.GgmlCuda);
        typeof(Qwen35Model).GetField("_fvDeviceStateCurrent", Instance)!.SetValue(model, deviceStateCurrent);
        return model;
    }

    private static void DrainForMigration(Qwen35Model model)
    {
        try { typeof(Qwen35Model).GetMethod("DrainFusedDecodeStateForMigration", Instance)!.Invoke(model, null); }
        catch (TargetInvocationException ex) when (ex.InnerException != null) { throw ex.InnerException; }
    }

    [Fact]
    public void TheMigration_DrainsAVerifyCurrentState_BeforeReadingTheMirrors()
    {
        var model = ModelWithVerifyCurrentState(deviceStateCurrent: true);

        DrainForMigration(model);

        Assert.Equal(1, model.Drains);
        Assert.False((bool)typeof(Qwen35Model).GetField("_fvDeviceStateCurrent", Instance)!.GetValue(model)!);
    }

    [Fact]
    public void WithTheMirrorsCurrent_TheMigrationDrainsNothing()
    {
        var model = ModelWithVerifyCurrentState(deviceStateCurrent: false);

        DrainForMigration(model);

        Assert.Equal(0, model.Drains);
    }
}
