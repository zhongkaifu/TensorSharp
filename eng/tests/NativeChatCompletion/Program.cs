// Exercise the compiled example's completion boundary without loading a model or changing engine APIs.
using System.Reflection;
using System.Runtime.Loader;

if (args.Length != 1) throw new ArgumentException("Pass the built TensorSharp.NativeChat.dll path.");
string app = Path.GetFullPath(args[0]);
string directory = Path.GetDirectoryName(app)!;
AssemblyLoadContext.Default.Resolving += (_, name) =>
{
    string candidate = Path.Combine(directory, name.Name + ".dll");
    return File.Exists(candidate) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(candidate) : null;
};
var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(app);
var chat = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(directory, "TensorSharp.Chat.dll"));
Type update = chat.GetType("TensorSharp.Server.ChatStreamUpdate", throwOnError: true)!;
MethodInfo require = assembly.GetType("NativeCompletion", throwOnError: true)!.GetMethod("Require")!;
object Terminal(string reason) => Activator.CreateInstance(update, "", true, 1, 1, 0, 0L, 0L, 0L, reason)!;
Exception? Invoke(object? terminal, bool hasText, CancellationToken cancellationToken)
{
    try { require.Invoke(null, [terminal, hasText, cancellationToken]); return null; }
    catch (TargetInvocationException e) { return e.InnerException; }
}
void Check(bool condition, string name)
{
    if (!condition) throw new InvalidOperationException("FAIL " + name);
    Console.WriteLine("PASS " + name);
}

using var cancelled = new CancellationTokenSource();
cancelled.Cancel();
Check(Invoke(Terminal("cancelled"), true, cancelled.Token) is OperationCanceledException,
    "locally cancelled token plus terminal cancelled follows cancellation path");
Check(Invoke(null, false, cancelled.Token) is OperationCanceledException,
    "locally cancelled token takes precedence over missing metrics/empty output");
Check(Invoke(Terminal("eos"), true, cancelled.Token) is OperationCanceledException,
    "deadline cancellation is preserved even if the stream reports eos");
Check(Invoke(Terminal("cancelled"), true, CancellationToken.None) is InvalidOperationException,
    "runtime cancellation without local cancellation remains an explicit generation error");
Check(Invoke(Terminal("eos"), true, CancellationToken.None) is null, "normal eos with text succeeds");
Check(Invoke(Terminal("eos"), false, CancellationToken.None) is InvalidOperationException,
    "empty completion fails");
Check(Invoke(null, true, CancellationToken.None) is InvalidOperationException,
    "missing terminal metrics fails");
Console.WriteLine("7 completion boundary checks passed; no model or native backend loaded.");
