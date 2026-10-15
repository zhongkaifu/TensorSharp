// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using TensorSharp.Memory.Hardware;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length is < 3 or > 6)
    throw new ArgumentException("Usage: GlmMemoryPlanProbe <model.gguf> <mmproj.gguf|none> <new-report.json> [context=32768] [prefill=128] [requests=3]");
string report = Path.GetFullPath(args[2]);
if (File.Exists(report)) throw new IOException("Use a new report path.");
int context = args.Length > 3 ? int.Parse(args[3], CultureInfo.InvariantCulture) : 32768;
int rows = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 128;
int requests = args.Length > 5 ? int.Parse(args[5], CultureInfo.InvariantCulture) : 3;
using var file = GgufFile.OpenForMemoryPlanning(args[0]);
var geometry = GlmUnifiedMemoryGeometry.Read(file);
var vision = args[1] == "none" ? null : GlmUnifiedVisionMemoryGeometry.Read(args[1]);
object? visionWeights = null;
if (vision != null)
{
    using var projector = GgufFile.OpenForMemoryPlanning(args[1]);
    long source = projector.Tensors.Values.Sum(t => projector.GetTensorFileRegion(t.Name).ByteLength);
    long decoded = projector.Tensors.Values.Sum(t => checked(t.NumElements * sizeof(float)));
    long alignment = checked(projector.Tensors.Count * (64L << 10));
    long extra = vision.DeviceWeightBytes - decoded - alignment;
    visionWeights = new { source_payload_bytes = source, source_file_bytes = new FileInfo(args[1]).Length,
        decoded_f32_weight_bytes = decoded, tensor_alignment_allowance_bytes = alignment,
        fold_merger_copy_allowance_bytes = extra, host_originals_and_transpose_bound_bytes = vision.HostWeightBytes,
        persistent_device_weight_bound_bytes = vision.DeviceWeightBytes, max_loading_temporary_bytes = vision.LoadingTemporaryBytes,
        tiny_image_execution_workspace = vision.ExecutionPeak(56, 56),
        device_upload_phase = "Lazily at first whole-encoder/merger graph; constructor dequantizes original host tensors and prepares patch/merger layouts." };
}
var solo = geometry.GraphPeak(context, 1, false);
var prefill = geometry.GraphPeak(context, rows, false);
var joint = geometry.GraphPeak(context, requests, requests > 1);
long former = checked((geometry.SlotBytes(context) + solo.Device) * requests
    + geometry.GraphPeak(Math.Min(context, 8192), rows, false).Device + joint.Device);
long corrected = geometry.ExecutionPlacementReserve(context, rows, requests);
var host = HostMemoryInfo.CaptureAllocationConstraints();
// nvidia-smi observes without initializing a TensorSharp/CUDA compute context.
// This is a planning snapshot, not allocator or physical-hard-cap qualification.
var start = new ProcessStartInfo("nvidia-smi") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
start.ArgumentList.Add("--query-gpu=memory.total,memory.free"); start.ArgumentList.Add("--format=csv,noheader,nounits");
using var process = Process.Start(start) ?? throw new IOException("Cannot observe NVIDIA memory.");
string output = process.StandardOutput.ReadToEnd(); string error = process.StandardError.ReadToEnd(); process.WaitForExit();
if (process.ExitCode != 0) throw new IOException("NVIDIA memory observation failed: " + error);
string[] first = output.Split('\n', StringSplitOptions.RemoveEmptyEntries)[0].Split(',');
long deviceTotal = checked(long.Parse(first[0].Trim(), CultureInfo.InvariantCulture) << 20);
long deviceFree = checked(long.Parse(first[1].Trim(), CultureInfo.InvariantCulture) << 20);
long headroom = Math.Max(768L << 20, deviceTotal / 16), ceiling = Math.Max(0, deviceFree - headroom);
object Placement(long execution, long? observedCeiling = null)
{
    try { return new { fits = true, cpu_expert_leading_layers = geometry.SelectCpuExpertLayers(observedCeiling ?? ceiling, execution, vision?.DeviceWeightBytes ?? 0), refusal = (string?)null }; }
    catch (Exception refused) when (refused is TensorSharp.Memory.MemoryPressureException or OverflowException)
    { return new { fits = false, cpu_expert_leading_layers = (int?)null, refusal = refused.Message }; }
}
Directory.CreateDirectory(Path.GetDirectoryName(report)!);
var evidence = new
{
    utc = DateTimeOffset.UtcNow, model = Path.GetFullPath(args[0]), mmproj = vision == null ? null : Path.GetFullPath(args[1]),
    architecture = geometry.Architecture, context, prefill_rows = rows, requests,
    dense_device_bytes = geometry.DenseDeviceBytes, expert_source_bytes = geometry.ExpertSourceBytes,
    vision_device_bytes = vision?.DeviceWeightBytes ?? 0, recurrent_state_per_slot_bytes = geometry.FixedStateBytes,
    vision_weights = visionWeights,
    full_slot_bytes = geometry.SlotBytes(context), independent_slots_bytes = checked(geometry.SlotBytes(context) * requests),
    graph_cache_entries = GlmDsaModel.UnifiedGraphCacheCapacity, replacement_generations = 2,
    solo_graph_peak = solo, prefill_graph_peak = prefill, joint_graph_peak = joint,
    former_execution_reserve_bytes = former, corrected_execution_reserve_bytes = corrected,
    recovered_device_allowance_bytes = former - corrected, former_placement = Placement(former), corrected_placement = Placement(corrected),
    hypothetical_fully_free_device_placement = new { ceiling_bytes = deviceTotal - headroom, former = Placement(former, deviceTotal - headroom), corrected = Placement(corrected, deviceTotal - headroom) },
    hardware = new { host, device_total_bytes = deviceTotal, device_free_bytes = deviceFree, device_headroom_bytes = headroom, device_ceiling_bytes = ceiling, nvidia_smi_raw = output },
    scope = "Bounded GGUF metadata and current hardware observation only; no model/device execution. Global-cache placement forecast preserves three independent full slots plus one cached graph and one replacement. nvidia-smi availability precedes TensorSharp CUDA context initialization. No empirical workspace, OS hard-cap, model quality or performance claim."
};
File.WriteAllText(report, JsonSerializer.Serialize(evidence, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(evidence));
