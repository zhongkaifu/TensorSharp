// Runs TensorSharp.Server.Host (or, with "cli" first, TensorSharp.Cli) unchanged, except that a Muse-Glimmer GGUF loads as the
// unified-runtime family (MuseGlimmerRuntimeModel) instead of the legacy model: the serving gates of the migration
// (docs/design/muse-glimmer-unified.md stage S3) run through the real surfaces before the descriptor switches (stage S8).
// See README.md.
using System.Reflection;
using TensorSharp.Models;

string surface = "server";
if (args.Length > 0 && args[0] is "server" or "cli")
{
    surface = args[0];
    args = args[1..];
}

// Before anything loads a model: the descriptor's factory reads it at every load.
MuseGlimmerArchitecture.DescriptorBuildsTheRuntimeFamily = true;
Console.Error.WriteLine($"[MuseGlimmerRuntimeHost] {surface}: a Muse-Glimmer GGUF loads as the unified-runtime family " +
    "(MuseGlimmerRuntimeModel), not the legacy model.");

Assembly host = Assembly.Load(surface == "cli" ? "TensorSharp.Cli" : "TensorSharp.Server.Host");
MethodInfo entry = host.EntryPoint ?? throw new InvalidOperationException($"{host.GetName().Name} has no entry point.");
object?[]? parameters = entry.GetParameters().Length == 0 ? null : new object?[] { args };
object? result = entry.Invoke(null, BindingFlags.DoNotWrapExceptions, binder: null, parameters, culture: null);
switch (result)
{
    case Task<int> withCode:
        return await withCode;
    case Task task:
        await task;
        break;
    case int code:
        return code;
}
return Environment.ExitCode;
