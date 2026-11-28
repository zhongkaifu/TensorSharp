// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Eng.GraphSchema;

bool check = Array.IndexOf(args, "--check") >= 0;
string root = FindRoot(AppContext.BaseDirectory) ?? FindRoot(Directory.GetCurrentDirectory())
    ?? throw new InvalidOperationException("Run from inside the TensorSharp checkout (no TensorSharp.slnx above here).");
GraphOpSchemaModel model = GraphOpSchemaGenerator.Parse(File.ReadAllText(Path.Combine(root, GraphOpSchemaGenerator.SchemaPath)));
int stale = 0;
foreach (KeyValuePair<string, string> file in GraphOpSchemaGenerator.Generate(model))
{
    string path = Path.Combine(root, file.Key);
    string? current = File.Exists(path) ? File.ReadAllText(path) : null;
    if (current == file.Value) { Console.WriteLine($"up to date  {file.Key}"); continue; }
    stale++;
    if (check) { Console.WriteLine($"STALE       {file.Key}"); continue; }
    File.WriteAllText(path, file.Value);
    Console.WriteLine($"generated   {file.Key}");
}
Console.WriteLine($"graph schema v{model.Version}, {model.Ops.Count} ops, hash {model.Hash}");
return check && stale > 0 ? 1 : 0;

static string? FindRoot(string start)
{
    for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
        if (File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx"))) return dir.FullName;
    return null;
}
