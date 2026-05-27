// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Text.Json;

namespace TensorSharp.TestMatrix.Configuration;

public static class MatrixConfigLoader
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    public static MatrixConfig LoadOrDefault(string? path)
    {
        string resolved = ResolveConfigPath(path);
        if (!File.Exists(resolved))
        {
            Console.Error.WriteLine($"[testmatrix] config file not found, using defaults: {resolved}");
            return new MatrixConfig();
        }

        try
        {
            string json = File.ReadAllText(resolved);
            MatrixConfig? cfg = JsonSerializer.Deserialize<MatrixConfig>(json, Options);
            if (cfg is null)
            {
                throw new InvalidDataException($"Config file deserialized to null: {resolved}");
            }
            Console.Error.WriteLine($"[testmatrix] loaded config: {resolved}");
            return cfg;
        }
        catch (Exception ex)
        {
            throw new InvalidDataException($"Failed to parse config file '{resolved}': {ex.Message}", ex);
        }
    }

    private static string ResolveConfigPath(string? path)
    {
        if (!string.IsNullOrWhiteSpace(path))
        {
            return Path.GetFullPath(path);
        }

        // Default: <bindir>/Defaults/matrix-config.json — copied next to the
        // assembly by the csproj None/CopyToOutputDirectory ItemGroup.
        string baseDir = AppContext.BaseDirectory;
        return Path.Combine(baseDir, "Defaults", "matrix-config.json");
    }
}
