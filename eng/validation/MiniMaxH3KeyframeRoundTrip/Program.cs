// How much of a keyframe survives MiniMax-H3's video VAE at a given canvas: the image is
// fitted to the canvas exactly as the pipeline fits a keyframe (centre crop, then Lanczos),
// encoded, decoded, and compared whole-frame. No denoiser runs, so this is the ceiling a
// pinned keyframe can be reproduced at. It is how the fl2v blurry-face report was traced:
// a full-body photo fitted to 640x384 round-tripped at 27.6 dB with its face smeared, and
// at 29.5 dB with the face intact on the 1152x768 recipe canvas.
//
//   dotnet run --project eng/validation/MiniMaxH3KeyframeRoundTrip -- \
//     --vae models/minimax_h3_video_vae_fp16.safetensors --backend metal \
//     --output artifacts/minimax-h3/keyframe-roundtrip --image photo.jpg [--image other.png] \
//     [--canvas 640x384 --canvas recipe]
//
// --canvas repeats; "recipe" is MiniMaxH3Geometry.KeyframeCanvas of each image, and the
// default is "recipe" plus 640x384. Evidence belongs in ignored artifacts/ or docs/validation/;
// never commit a personal photo or its reconstruction. Faces are judged by eye on the saved
// PNGs: the PSNR is over the whole frame.
using System.Globalization;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

List<string> Values(string name) => args.Select((value, i) => (value, i)).Where(p => p.value == name && p.i + 1 < args.Length).Select(p => args[p.i + 1]).ToList();
string vae = Path.GetFullPath(Values("--vae").SingleOrDefault() ?? throw new ArgumentException("--vae <minimax_h3_video_vae_fp16.safetensors> required"));
string output = Path.GetFullPath(Values("--output").SingleOrDefault() ?? throw new ArgumentException("--output <dir under artifacts/ or docs/validation/> required"));
List<string> images = Values("--image").Select(Path.GetFullPath).ToList();
if (images.Count == 0) throw new ArgumentException("at least one --image required");
List<string> canvases = Values("--canvas");
if (canvases.Count == 0) canvases = ["recipe", "640x384"];
string backend = (Values("--backend").SingleOrDefault() ?? "metal").ToLowerInvariant();
var (ggml, type) = backend switch
{
    "metal" => (GgmlBackendType.Metal, BackendType.GgmlMetal),
    "cuda" => (GgmlBackendType.Cuda, BackendType.GgmlCuda),
    "cpu" => (GgmlBackendType.Cpu, BackendType.GgmlCpu),
    _ => throw new ArgumentException("--backend must be metal, cuda or cpu"),
};
Directory.CreateDirectory(output);

GgmlBasicOps.EnsureBackendAvailable(ggml);
using var model = new MiniMaxH3VideoVae(vae, type, null!);
var records = new List<object>();
foreach (string path in images)
{
    RgbImage source = ImageIO.Load(path); // EXIF orientation applied, as the pipeline decodes keyframes
    foreach (string canvas in canvases)
    {
        var (w, h) = canvas == "recipe" ? MiniMaxH3Geometry.KeyframeCanvas(source.Width, source.Height) : Parse(canvas);
        RgbImage fitted = ImageIO.ResizeCover(source, w, h);
        float[] latent = model.EncodeFrame(fitted);
        float[] pixels = model.Decode(latent, 1, h / MiniMaxH3Geometry.VaeSpatialRatio, w / MiniMaxH3Geometry.VaeSpatialRatio);
        MiniMaxH3VideoVae.ToRgb(pixels, MiniMaxH3VideoVae.DecodedFrameCount(1), h, w);
        long plane = (long)w * h;
        var chw = new float[3 * plane];
        for (int c = 0; c < 3; c++) Array.Copy(pixels, c * plane, chw, c * plane, plane);
        RgbImage decoded = RgbImage.FromPlanarChw(w, h, chw);
        double squared = 0;
        for (long i = 0; i < fitted.Pixels.Length; i++)
        {
            double a = Math.Round(fitted.Pixels[i] * 255), b = Math.Round(decoded.Pixels[i] * 255);
            squared += (a - b) * (a - b);
        }
        double psnr = 10 * Math.Log10(255.0 * 255.0 / (squared / fitted.Pixels.Length));
        string stem = $"{Path.GetFileNameWithoutExtension(path)}_{w}x{h}";
        ImageIO.SavePng(Path.Combine(output, stem + "_fitted.png"), fitted);
        ImageIO.SavePng(Path.Combine(output, stem + "_vae.png"), decoded);
        Console.WriteLine($"{Path.GetFileName(path)} {source.Width}x{source.Height} -> {w}x{h} ({canvas}): whole-frame PSNR {psnr:F2} dB");
        records.Add(new { image = Path.GetFileName(path), source = new[] { source.Width, source.Height }, canvas, size = new[] { w, h }, psnrDb = Math.Round(psnr, 2) });
    }
}
File.WriteAllText(Path.Combine(output, "roundtrip.json"), JsonSerializer.Serialize(new
{
    backend, vae = Path.GetFileName(vae), scope = "VAE encode then decode of a centre-cropped keyframe; no denoiser; whole-frame PSNR",
    records,
}, new JsonSerializerOptions { WriteIndented = true }) + "\n");

static (int, int) Parse(string canvas)
{
    string[] parts = canvas.Split('x');
    if (parts.Length != 2 || !int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int w)
        || !int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int h)
        || w <= 0 || h <= 0 || w % MiniMaxH3Geometry.SpatialMultiple != 0 || h % MiniMaxH3Geometry.SpatialMultiple != 0)
        throw new ArgumentException($"--canvas '{canvas}' must be 'recipe' or WxH in multiples of {MiniMaxH3Geometry.SpatialMultiple}");
    return (w, h);
}
