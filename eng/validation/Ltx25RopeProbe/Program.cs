using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Ltx25;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
    if (i + 1 == args.Length || !args[i].StartsWith("--") || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Expected unique --option value pairs.");
string capturePath = Path.GetFullPath(options["capture"]), output = Path.GetFullPath(options["output"]);
if (File.Exists(output)) throw new IOException("Refusing to overwrite evidence.");
using var document = JsonDocument.Parse(File.ReadAllBytes(capturePath));
var capture = document.RootElement;
if (capture.GetProperty("input_mode").GetString() != "captured") throw new ArgumentException("Expected captured inputs.");
string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
float[] Read(string name)
{
    var spec = capture.GetProperty("inputs").GetProperty(name);
    if (spec.GetProperty("dtype").GetString() != "float32-le" || !BitConverter.IsLittleEndian)
        throw new ArgumentException("Expected little-endian Float32.");
    string path = spec.GetProperty("path").GetString()!;
    if (!Path.IsPathRooted(path)) path = Path.Combine(Path.GetDirectoryName(capturePath)!, path);
    byte[] bytes = File.ReadAllBytes(path);
    int count = spec.GetProperty("shape").EnumerateArray().Aggregate(1, (n, value) => checked(n * value.GetInt32()));
    if (bytes.Length != checked(count * 4) || Hash(bytes) != spec.GetProperty("sha256").GetString())
        throw new InvalidDataException("Captured array identity differs: " + name);
    return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
}
var config = capture.GetProperty("config");
var tokens = capture.GetProperty("tokens");
var precision = Enum.Parse<Ltx25FrequencyPrecision>(capture.GetProperty("frequency_precision").GetString()!);
if (precision != Ltx25FrequencyPrecision.Float64) throw new ArgumentException("This phase diagnostic currently supports the captured Float64 frequency mode.");
var videoPositions = new Ltx25PositionGrid(tokens.GetProperty("video").GetInt32(), 3, Read("video_positions"));
var audioPositions = new Ltx25PositionGrid(tokens.GetProperty("audio").GetInt32(), 1, Read("audio_positions"));
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var streams = new Dictionary<string, object>();
void Generate(string name, Ltx25PositionGrid positions, int dimensions, int heads)
{
    var rope = Ltx25Geometry.BuildRope(positions, dimensions, heads, precision);
    object Save(string component, float[] values, int[]? shape = null)
    {
        string path = output + "." + name + "." + component + ".f32";
        using (var file = new FileStream(path, FileMode.CreateNew)) file.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        using var input = File.OpenRead(path);
        return new { path, sha256 = Convert.ToHexStringLower(SHA256.HashData(input)), shape = shape ?? new[] { positions.Tokens, dimensions / 2 } };
    }
    // Independently expose the intermediate scalar arithmetic from BuildRope so
    // a mismatch can be localized before trigonometry. Production is unchanged;
    // cosine/sine above still come from the actual method, not this expansion.
    int half = dimensions / 2, count = dimensions / (2 * positions.Axes), pad = half - count * positions.Axes;
    var frequencies = Enumerable.Range(0, count).Select(i => (float)(Math.Pow(10000f, count == 1 ? 0 : (double)i / (count - 1)) * (Math.PI / 2))).ToArray();
    var coordinates = new float[positions.Tokens * positions.Axes];
    var phases = new float[positions.Tokens * half];
    for (int token = 0; token < positions.Tokens; token++)
    for (int axis = 0; axis < positions.Axes; axis++)
    {
        int coordinateOffset = (token * positions.Axes + axis) * 2;
        float midpoint = (positions.Boundaries[coordinateOffset] + positions.Boundaries[coordinateOffset + 1]) * .5f;
        float coordinate = midpoint / (axis == 0 ? 20f : 2048f) * 2f - 1f;
        coordinates[token * positions.Axes + axis] = coordinate;
        for (int index = 0; index < count; index++)
            phases[token * half + pad + index * positions.Axes + axis] = frequencies[index] * coordinate;
    }
    streams[name] = new { dimensions, heads, axes = positions.Axes, cosine = Save("cosine", rope.Cosine), sine = Save("sine", rope.Sine),
        frequency_grid = Save("frequency-grid", frequencies, [count]), coordinates = Save("coordinates", coordinates, [positions.Tokens, positions.Axes]),
        phases = Save("phases", phases) };
}
Generate("video", videoPositions, config.GetProperty("VideoDim").GetInt32(), config.GetProperty("VideoHeads").GetInt32());
Generate("audio", audioPositions, config.GetProperty("AudioDim").GetInt32(), config.GetProperty("AudioHeads").GetInt32());
Generate("cross_video", Ltx25Geometry.TemporalPositions(videoPositions), config.GetProperty("AudioDim").GetInt32(), config.GetProperty("AudioHeads").GetInt32());
var report = new { completed = true, capture = capturePath, capture_sha256 = Hash(File.ReadAllBytes(capturePath)),
    frequency_precision = precision.ToString(), runtime = RuntimeInformation.FrameworkDescription,
    source = "TensorSharp.Models/Models/Ltx25/Ltx25Geometry.cs", streams,
    scope = "CPU-only actual Ltx25Geometry.BuildRope compiled directly from production source; no model weights or native/GPU backend." };
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Console.WriteLine(output);
