// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorAgent.Core.Settings;
using TensorSharp.Runtime.Speculative;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// Speculative decoding for the model the app is about to use.
///
/// <para>
/// A draft head guesses a few tokens ahead and the trunk verifies them in ONE
/// batched forward; an n-gram drafter does the same with no weights at all, from
/// the tokens the conversation already holds. Both leave the output exactly what
/// plain decoding would have produced - a wrong guess costs a rollback, never a
/// token - and the engine's cost governor parks drafting while it measures as a
/// loss. Measured on the Mac (ggml_metal, greedy, plain → speculative): Gemma 4
/// E4B with its draft head 46 → 92 tok/s; Qwen 3.5-9B with n-gram 31 → 86 tok/s
/// on an answer that quotes a file, and break-even on prose (Gemma 4 E2B 81.5 →
/// 80.9). Agent turns quote files, tool results and their own earlier answers,
/// which is where a lookup drafter pays.
/// </para>
/// <para>
/// The catalog lists the draft head as an optional companion file, and it used
/// to be downloaded by nothing and attached by nothing. Now it is fetched with
/// the other optional files and handed to the engine the same way the CLI's
/// <c>--draft-model</c> is, through the environment the loader reads. The
/// algorithm is chosen AFTER the load: <c>auto</c> only when a draft head really
/// attached (<c>auto</c> with no head declines outright), n-gram otherwise.
/// </para>
/// <para>
/// Everything here goes through the environment because that is the seam
/// <c>SchedulerConfig.FromEnvironment</c> and <c>SpeculativeDraftHeadLoader</c>
/// already read, and they read it when the engine is constructed - after this.
/// A developer's own <c>TS_SPEC</c> / <c>TS_SPEC_TYPE</c> in the launch
/// environment wins over the setting, exactly like the prefill chunk does.
/// </para>
/// </summary>
public static class SpeculationPolicy
{
    public const string EnabledVariable = SpeculationEnvVars.Enabled;
    public const string TypeVariable = SpeculationEnvVars.Type;
    public const string DraftModelVariable = SpeculationEnvVars.DraftModel;

    // What the process was launched with, recorded before the app writes anything.
    // An explicit static constructor: a field initializer alone runs at the first
    // static-field ACCESS, which inside PrepareLoad comes after the policy's own
    // write of TS_SPEC, so it recorded that write as the launch environment.
    private static string? LaunchEnabled;
    private static string? LaunchType;

    static SpeculationPolicy()
    {
        LaunchEnabled = Environment.GetEnvironmentVariable(EnabledVariable);
        LaunchType = Environment.GetEnvironmentVariable(TypeVariable);
    }

    /// <summary>Pin what counts as the launch environment - for a test process, whose
    /// first touch of this type may come after another test set the variables.</summary>
    internal static void UseLaunchEnvironment(string? enabled, string? type)
    {
        LaunchEnabled = enabled;
        LaunchType = type;
    }

    /// <summary>
    /// Before the load: name the draft head the loader should attach (or clear a
    /// stale one from the previous model) and switch speculation on or off per the
    /// setting. Returns a one-line account for the log.
    /// </summary>
    public static string PrepareLoad(AppSettings? settings, string? draftHeadPath)
        => PrepareLoad(settings, draftHeadPath, LaunchEnabled);

    internal static string PrepareLoad(AppSettings? settings, string? draftHeadPath, string? launchEnabled)
    {
        bool enabled = settings?.SpeculativeDecoding ?? true;
        Environment.SetEnvironmentVariable(DraftModelVariable,
            string.IsNullOrWhiteSpace(draftHeadPath) ? null : draftHeadPath);
        string source;
        if (string.IsNullOrWhiteSpace(launchEnabled))
        {
            Environment.SetEnvironmentVariable(EnabledVariable, enabled ? "1" : "0");
            // Provisionally the weight-free algorithm, because it is the one that
            // needs no draft head; ChooseAlgorithm upgrades it once a head has really
            // attached. It does not serve every model: n-gram still needs a target
            // that can verify a draft window in one pass (not Qwen 3 / Bonsai 8B,
            // GPT-OSS or Mistral 3, and Nemotron-H refuses every speculator). On those
            // the engine's capability report and execution plan log the reason and
            // every reply decodes plainly -- a no-op, not an error.
            if (string.IsNullOrWhiteSpace(LaunchType))
                Environment.SetEnvironmentVariable(TypeVariable, SpeculatorRegistry.NGram);
            source = enabled ? "on (setting)" : "off (setting)";
        }
        else
        {
            source = $"{EnabledVariable}={launchEnabled} from the launch environment";
        }
        return $"speculative decoding {source}; draft head {(draftHeadPath is { Length: > 0 } ? Path.GetFileName(draftHeadPath) : "none")}";
    }

    /// <summary>
    /// After the load, before the engine is built: <c>auto</c> when a draft head is
    /// attached, n-gram otherwise. Returns the algorithm name that will apply.
    /// </summary>
    public static string ChooseAlgorithm(bool draftHeadAttached)
        => ChooseAlgorithm(draftHeadAttached, LaunchType);

    internal static string ChooseAlgorithm(bool draftHeadAttached, string? launchType)
    {
        if (!string.IsNullOrWhiteSpace(launchType))
            return launchType.Trim();
        string algorithm = draftHeadAttached ? SpeculatorRegistry.Auto : SpeculatorRegistry.NGram;
        Environment.SetEnvironmentVariable(TypeVariable, algorithm);
        return algorithm;
    }
}
