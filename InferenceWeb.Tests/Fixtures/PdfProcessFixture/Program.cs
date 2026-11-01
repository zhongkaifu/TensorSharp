using System.Diagnostics;
using System.Reflection;
using System.Text.Json;

switch (args[0])
{
    case "arguments":
        Console.Write(JsonSerializer.Serialize(args.Skip(1).ToArray()));
        break;
    case "environment":
        Console.Write(Environment.GetEnvironmentVariable(args[1]));
        break;
    case "duplex":
        Task errors = Console.OpenStandardError().WriteAsync(new byte[96 * 1024]).AsTask();
        await Console.OpenStandardInput().CopyToAsync(Console.OpenStandardOutput());
        await errors;
        break;
    case "overflow":
        PublishPidFile(args[1], writer => writer.Write(Environment.ProcessId));
        await Console.OpenStandardOutput().WriteAsync(new byte[int.Parse(args[2])]);
        await Task.Delay(Timeout.Infinite);
        break;
    case "tree":
        var start = new ProcessStartInfo(Environment.ProcessPath!) { UseShellExecute = false };
        start.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        start.ArgumentList.Add("sleep");
        using (Process child = Process.Start(start)!)
        {
            PublishPidFile(args[1], writer => writer.Write($"{Environment.ProcessId},{child.Id}"));
            await Task.Delay(Timeout.Infinite);
        }
        break;
    case "publish-pids":
        // Deterministically expose both incomplete states to the readiness test.
        // The final PID path must remain absent until this writer is closed.
        PublishPidFile(args[1], writer =>
        {
            AwaitPublicationRelease(args[1], "empty");
            writer.Write("123,");
            writer.Flush();
            AwaitPublicationRelease(args[1], "partial");
            writer.Write("456");
        });
        break;
    case "sleep":
        await Task.Delay(Timeout.Infinite);
        break;
    case "reject":
        Console.OpenStandardInput().Dispose();
        Console.Error.Write("fixture rejected request");
        return 17;
    case "invalid-utf8":
        await Console.OpenStandardError().WriteAsync(new byte[] { 255, 254, 0x61 });
        break;
    default:
        return 19;
}
return 0;

static void PublishPidFile(string path, Action<TextWriter> write)
{
    // File existence is the reader's readiness signal. Publish only after the
    // complete contents are closed, using a same-directory atomic rename.
    string pending = path + ".pending";
    using (var writer = new StreamWriter(new FileStream(pending, FileMode.CreateNew, FileAccess.Write, FileShare.None)))
        write(writer);
    File.Move(pending, path);
}

static void AwaitPublicationRelease(string path, string stage)
{
    File.WriteAllText(path + "." + stage, "ready");
    while (!File.Exists(path + "." + stage + ".release"))
        Thread.Sleep(25);
}
