// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.TestMatrix.Reporting;

/// <summary>
/// Persists per-case <see cref="TestResult"/> JSON under a results directory
/// (one file per case) and reloads them for report generation.
/// </summary>
public sealed class ResultStore
{
    private static readonly JsonSerializerOptions WriteOptions = new()
    {
        WriteIndented = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private static readonly JsonSerializerOptions ReadOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    private readonly string _resultsDir;

    public ResultStore(string resultsDir)
    {
        _resultsDir = resultsDir;
        Directory.CreateDirectory(_resultsDir);
    }

    public string PathFor(string caseId)
    {
        // Sanitize: case ids contain '=' from env-var sweeps. Replace with '@'
        // to keep filenames readable on every OS.
        string safe = caseId.Replace('=', '@');
        return Path.Combine(_resultsDir, safe + ".json");
    }

    public bool Exists(string caseId) => File.Exists(PathFor(caseId));

    public void Save(TestResult result)
    {
        string path = PathFor(result.CaseId);
        string json = JsonSerializer.Serialize(result, WriteOptions);
        File.WriteAllText(path, json);
    }

    public IReadOnlyList<TestResult> LoadAll()
    {
        var list = new List<TestResult>();
        if (!Directory.Exists(_resultsDir))
        {
            return list;
        }
        foreach (string file in Directory.EnumerateFiles(_resultsDir, "*.json", SearchOption.TopDirectoryOnly))
        {
            try
            {
                TestResult? r = JsonSerializer.Deserialize<TestResult>(File.ReadAllText(file), ReadOptions);
                if (r is not null)
                {
                    list.Add(r);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[testmatrix] skip unreadable result file {file}: {ex.Message}");
            }
        }
        return list;
    }
}
