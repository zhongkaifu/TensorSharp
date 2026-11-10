using System;
using System.Collections.Generic;
using System.Globalization;

namespace TensorSharp.Distributed;

/// <summary>Shared CLI/server selection of tensor parallelism or whole-layer placement.</summary>
public sealed class ModelParallelismOptions
{
    public const string TpDegreeVariable = "TENSORSHARP_TP_DEGREE";
    public const string LayerSplitDegreeVariable = "TENSORSHARP_LAYER_SPLIT_DEGREE";
    public const string NodeIdVariable = "TENSORSHARP_TP_NODE_ID";
    public const string PeersVariable = "TENSORSHARP_TP_PEERS";

    private readonly Dictionary<string, string> _overrides = new();
    public int TpDegree { get; private set; }
    public int LayerSplitDegree { get; private set; }
    /// <summary>Whole-layer execution was selected, including one GPU per node.</summary>
    public bool IsLayerSplit { get; private set; }
    public DistributedTpConfig Distributed { get; private set; }
    public bool HasCliOverrides => _overrides.Count != 0;

    /// <summary>Collect one placement option at the host parser's current option
    /// boundary. The host must consume other options' operands itself so literal
    /// prompt text such as "--tp=2" is never interpreted as configuration.</summary>
    public static bool TryCollect(string[] args, ref int index, List<string> destination)
    {
        string arg = args[index];
        int equals = arg.IndexOf('=');
        string flag = equals < 0 ? arg : arg[..equals];
        if (VariableFor(flag) == null) return false;

        string value;
        if (equals >= 0)
            value = arg[(equals + 1)..];
        else if (index + 1 < args.Length && !args[index + 1].StartsWith("--", StringComparison.Ordinal))
            value = args[++index];
        else
            throw new ArgumentException($"{flag} requires a value.");
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"{flag} requires a value.");
        destination?.Add(flag + "=" + value);
        return true;
    }

    private static string VariableFor(string flag) => flag.ToLowerInvariant() switch
    {
        "--tp" => TpDegreeVariable,
        "--layer-split" => LayerSplitDegreeVariable,
        "--tp-node-id" => NodeIdVariable,
        "--tp-peers" => PeersVariable,
        _ => null,
    };

    /// <summary>Validate options already collected by the host parser before
    /// publishing environment changes. Hosts should call TryCollect from their
    /// argument loop rather than passing raw arguments containing other options' values.</summary>
    public static ModelParallelismOptions Parse(string[] args, List<string> remaining = null)
    {
        var result = new ModelParallelismOptions();
        args ??= Array.Empty<string>();
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            int equals = arg.IndexOf('=');
            string flag = equals < 0 ? arg : arg[..equals];
            string variable = VariableFor(flag);
            if (variable == null)
            {
                remaining?.Add(arg);
                continue;
            }

            string value;
            if (equals >= 0)
                value = arg[(equals + 1)..];
            else if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                value = args[++i];
            else
                throw new ArgumentException($"{flag} requires a value.");
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException($"{flag} requires a value.");
            // Validate each occurrence, including one superseded by a later value.
            if (variable != PeersVariable)
                ReadInteger(value, flag, variable == NodeIdVariable ? 0 : 1);
            result._overrides[variable] = value;
        }

        string Resolve(string variable) => result._overrides.TryGetValue(variable, out var value)
            ? value : Environment.GetEnvironmentVariable(variable);
        result.TpDegree = ReadInteger(Resolve(TpDegreeVariable) ?? "1", "--tp / " + TpDegreeVariable, 1);
        result.LayerSplitDegree = ReadInteger(Resolve(LayerSplitDegreeVariable) ?? "1",
            "--layer-split / " + LayerSplitDegreeVariable, 1);
        // Both above 1 is hybrid placement: --layer-split stages, each a --tp group, in one
        // process (whether the architecture supports it is the loader's decision, from its
        // serving contract). IsLayerSplit names the PURE whole-layer mode.
        result.IsLayerSplit = result.TpDegree == 1 && !string.IsNullOrWhiteSpace(Resolve(LayerSplitDegreeVariable));

        string node = Resolve(NodeIdVariable);
        string peers = Resolve(PeersVariable);
        bool hasNode = !string.IsNullOrWhiteSpace(node);
        bool hasPeers = !string.IsNullOrWhiteSpace(peers);
        if (hasNode && result.TpDegree > 1 && result.LayerSplitDegree > 1)
            throw new ArgumentException(
                "--tp with --layer-split (hybrid placement) runs in one process; it cannot be combined with --tp-node-id/--tp-peers.");
        if (hasNode != hasPeers)
            throw new ArgumentException(hasNode
                ? "--tp-node-id requires --tp-peers (the endpoints of all nodes)."
                : "--tp-peers requires --tp-node-id (this node's 0-based ID).");
        if (hasNode)
        {
            int nodeId = ReadInteger(node, "--tp-node-id / " + NodeIdVariable, 0);
            try
            {
                var endpoints = DistributedTpConfig.ParsePeers(peers);
                if (endpoints.Length < 2)
                    throw new ArgumentException("Distributed inference requires at least two peer endpoints.");
                if (nodeId >= endpoints.Length)
                    throw new ArgumentException($"--tp-node-id {nodeId} must be less than the number of peers ({endpoints.Length}).");
                result.Distributed = new DistributedTpConfig(nodeId,
                    result.IsLayerSplit ? result.LayerSplitDegree : result.TpDegree, endpoints);
            }
            catch (Exception ex) when (ex is ArgumentException or FormatException or System.Net.Sockets.SocketException)
            {
                throw new ArgumentException($"Invalid --tp-peers / --tp-node-id configuration: {ex.Message}", ex);
            }
        }
        return result;
    }

    public bool ApplyEnvironment()
    {
        foreach (var pair in _overrides)
            Environment.SetEnvironmentVariable(pair.Key, pair.Value);
        return HasCliOverrides;
    }

    private static int ReadInteger(string value, string flag, int minimum)
    {
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int number) || number < minimum)
            throw new ArgumentException($"Invalid value for {flag}: '{value}'. Expected an integer >= {minimum}.");
        return number;
    }
}
