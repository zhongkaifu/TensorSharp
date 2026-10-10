// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The cost of keying a Qwen-Image-2.1 edit's noise to its references: the SHA-256 over the
// pictures (QwenImage21Sampling.ReferenceStream) and, for scale, the Philox draw every request
// already makes. Synthetic pictures; no weights, encoder, transformer, VAE or device are touched.
//
//   dotnet run -c Release --project eng/validation/QwenImage21EditNoiseCost -- [--sizes 1024x1024,2048x2048,4032x3024] [--reps 7]
//   dotnet run -c Release --project eng/validation/QwenImage21EditNoiseCost -p:ModelsDir=TensorSharp.Cli/bin
//
// The second form measures the TensorSharp.Models.dll in ModelsDir as it was built (the CLI's
// output is a Debug build). Each size is hashed with no alpha plane, an all-opaque one (hashed as
// none, after a scan) and one translucent pixel at the end (the scan's worst case, then the alpha
// plane is hashed too). Prints the runtime it ran on: the numbers hold for that runtime only.
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.Models.QwenImage;

string Option(string name, string fallback)
{
    int index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
}

(int Width, int Height)[] sizes = Option("--sizes", "1024x1024,2048x2048,4032x3024").Split(',')
    .Select(size => size.Split('x'))
    .Select(sides => (int.Parse(sides[0]), int.Parse(sides[1])))
    .ToArray();
int repetitions = int.Parse(Option("--reps", "7"));
if (repetitions < 1 || sizes.Any(s => s.Width < 1 || s.Height < 1))
    throw new ArgumentException("Repetitions and sides must be positive.");

// Internal API, by reflection: the same code works against a DLL built elsewhere (ModelsDir).
Type sampling = typeof(RgbImage).Assembly.GetType("TensorSharp.Models.QwenImage.QwenImage21Sampling", throwOnError: true)!;
MethodInfo referenceStream = sampling.GetMethod("ReferenceStream", BindingFlags.Static | BindingFlags.NonPublic)
    ?? throw new MissingMethodException("QwenImage21Sampling.ReferenceStream: this TensorSharp.Models predates edit noise.");
MethodInfo noise = sampling.GetMethod("Noise", BindingFlags.Static | BindingFlags.NonPublic, new[] { typeof(int), typeof(long), typeof(ulong) })
    ?? throw new MissingMethodException("QwenImage21Sampling.Noise(int, long, ulong)");
ulong Stream(RgbImage image) => (ulong)referenceStream.Invoke(null, new object[] { (IReadOnlyList<RgbImage>)new[] { image } })!;

static RgbImage Picture(int width, int height, string alpha)
{
    var random = new Random(1);
    var pixels = new float[checked(width * height * 3)];
    for (int i = 0; i < pixels.Length; i++) pixels[i] = random.Next(256) / 255f;
    float[]? plane = alpha switch
    {
        "none" => null,
        "opaque" => Enumerable.Repeat(1f, width * height).ToArray(),
        _ => Enumerable.Range(0, width * height).Select(i => i == width * height - 1 ? 0.5f : 1f).ToArray(),
    };
    return new RgbImage(width, height, pixels, plane!);
}

double Median(Action action)
{
    action();
    var times = new List<double>();
    for (int r = 0; r < repetitions; r++)
    {
        var watch = Stopwatch.StartNew();
        action();
        times.Add(watch.Elapsed.TotalMilliseconds);
    }
    times.Sort();
    return times[times.Count / 2];
}

Console.WriteLine($"runtime {RuntimeInformation.FrameworkDescription} ({(Type.GetType("Mono.Runtime") != null ? "Mono" : "CoreCLR")}), " +
    $"{RuntimeInformation.ProcessArchitecture}, {Environment.ProcessorCount} cores; " +
    $"TensorSharp.Models {typeof(RgbImage).Assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration ?? "?"} " +
    $"from {Path.GetDirectoryName(typeof(RgbImage).Assembly.Location)}");
foreach (var (width, height) in sizes)
    foreach (string alpha in new[] { "none", "opaque", "translucent" })
    {
        var image = Picture(width, height, alpha);
        Console.WriteLine($"hash {width}x{height} ({width * (double)height / 1e6:F1} MP) alpha={alpha,-11} median {Median(() => Stream(image)),8:F1} ms");
    }
foreach (int side in new[] { 1024, 2048 })
{
    int count = side / 16 * (side / 16) * 64;
    Console.WriteLine($"noise draw for a {side}x{side} latent ({count} values): median {Median(() => noise.Invoke(null, new object[] { count, 0L, 0x1234UL })),8:F1} ms");
}
