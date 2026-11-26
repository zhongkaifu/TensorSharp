using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;

// The full-depth MiniMax-H3 denoiser (MiniMaxH3Denoiser: the diffusion runtime's step program) on one backend over exact
// saved inputs, for the independent F32 reference (minimax-h3-full-dit-reference.py). --backend cuda (the default) runs the
// direct CUDA tensor executor; ggml_cuda, ggml_cpu and cpu run the GGML graph executor or the managed one.
string? Value(string name) => args.SkipWhile(value=>value!=name).Skip(1).FirstOrDefault();
string backendName=Value("--backend") ?? "cuda";
BackendType backend=backendName switch
{
    "cuda" => BackendType.Cuda, "ggml_cuda" => BackendType.GgmlCuda, "ggml_cpu" => BackendType.GgmlCpu,
    "ggml_metal" => BackendType.GgmlMetal, "cpu" => BackendType.Cpu,
    _ => throw new ArgumentException($"--backend {backendName}: cuda, ggml_cuda, ggml_cpu, ggml_metal or cpu"),
};
string model=Value("--model") ?? throw new ArgumentException("--model is required");
string? reference=Value("--reference"),fixtures=Value("--fixtures");
if((reference==null)==(fixtures==null)) throw new ArgumentException("Specify exactly one of --reference or --fixtures");
string inputRoot=Path.GetFullPath(reference ?? fixtures!);
string inputMode=fixtures==null ? "sd-callback" : "synthetic-fixtures";
string output=Path.GetFullPath(Value("--output") ?? throw new ArgumentException("--output is required"));
float sigma=float.Parse(Value("--sigma") ?? "1",System.Globalization.CultureInfo.InvariantCulture);
if(fixtures!=null && sigma!=1) throw new ArgumentException("Full50 fixture validation requires sigma one");
bool traceEnabled=!args.Contains("--no-trace");
int warmup=int.Parse(Value("--warmup") ?? "0"),repeats=int.Parse(Value("--repeats") ?? "1");
if(warmup<0 || repeats<1 || (traceEnabled && warmup+repeats!=1)) throw new ArgumentException("Repeated timings require --no-trace");
string trace=Path.GetFullPath(Value("--trace-dir") ?? Path.Combine(Path.GetDirectoryName(output)!,"trace"));
if(File.Exists(output) || File.Exists(Path.Combine(trace,"packed-in.f32")) ||
   File.Exists(Path.Combine(trace,"carrier-audio-velocity.f32"))) throw new InvalidOperationException("Refusing existing trace output");
Directory.CreateDirectory(trace);
Environment.SetEnvironmentVariable("TS_H3_LINEAR_TRACE_FILE",traceEnabled && backend == BackendType.Cuda ? Path.Combine(trace,"linear-calls.jsonl") : null);
var metadata=Directory.GetFiles(inputRoot,"*.f32.json")
    .Select(path=>JsonSerializer.Deserialize<JsonElement>(File.ReadAllText(path))).ToArray();
float[] Read(string file)
{
    byte[] bytes=File.ReadAllBytes(Path.Combine(inputRoot,file));
    if(bytes.Length%4!=0) throw new InvalidDataException("F32 file is not aligned");
    var values=new float[bytes.Length/4]; Buffer.BlockCopy(bytes,0,values,0,bytes.Length);
    if(values.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite reference input");
    return values;
}
JsonElement ProjectionInput(string suffix)
{
    var projection=metadata.Single(item=>item.GetProperty("file").GetString()!.StartsWith("projection-") &&
        item.GetProperty("sources").EnumerateArray().Any(source=>source.GetProperty("name").GetString()!.Contains(suffix)));
    int id=projection.GetProperty("sources")[1].GetProperty("id").GetInt32();
    return metadata.Single(item=>item.GetProperty("id").GetInt32()==id);
}
int latentW,latentH,latentT,audioCount,textCount;
float[] video,audio,text;
string[] inputFiles;
if(fixtures!=null)
{
    (float[] Values,int[] Shape) Fixture(string file,int rank)
    {
        int[] dimensions=File.ReadAllText(Path.Combine(inputRoot,file+".shape")).Trim().Split(',').Select(int.Parse).ToArray();
        if(dimensions.Length!=rank || dimensions.Any(value=>value<=0)) throw new InvalidDataException("Invalid fixture shape: "+file);
        float[] values=Read(file);
        if(dimensions.Aggregate(1L,(size,value)=>checked(size*value))!=values.LongLength)
            throw new InvalidDataException("Fixture size differs from shape: "+file);
        return(values,dimensions);
    }
    var vf=Fixture("h3_dit_video_latent.bin",4);
    var af=Fixture("h3_dit_audio_latent.bin",3);
    var tf=Fixture("h3_dit_text_hidden.bin",2);
    if(vf.Shape[3]!=24 || vf.Shape[1]%2!=0 || vf.Shape[2]%2!=0 || af.Shape[0]!=2 || af.Shape[2]!=32)
        throw new InvalidDataException("Unsupported fixture channels or spatial patch dimensions");
    latentT=vf.Shape[0]; latentH=vf.Shape[1]; latentW=vf.Shape[2];
    audioCount=checked(2*af.Shape[1]); textCount=tf.Shape[0];
    video=MiniMaxH3Latents.PatchifyVideo(vf.Values,latentT,latentH,latentW,24);
    audio=af.Values; text=tf.Values;
    inputFiles=new[]{"h3_dit_video_latent.bin","h3_dit_audio_latent.bin","h3_dit_text_hidden.bin"}
        .SelectMany(file=>new[]{file,file+".shape"}).ToArray();
}
else
{
    var vInput=ProjectionInput("video_patch_proj.weight");
    var aInput=ProjectionInput("audio_patch_proj.weight");
    var tInput=ProjectionInput("condition_proj.weight");
    var videoLeaf=metadata.Single(item=>item.GetProperty("file").GetString()!.StartsWith("input-") &&
        item.GetProperty("ne")[3].GetInt32()==24);
    latentW=videoLeaf.GetProperty("ne")[0].GetInt32(); latentH=videoLeaf.GetProperty("ne")[1].GetInt32();
    latentT=videoLeaf.GetProperty("ne")[2].GetInt32();
    audioCount=aInput.GetProperty("ne")[1].GetInt32(); textCount=tInput.GetProperty("ne")[1].GetInt32();
    inputFiles=new[]{vInput,aInput,tInput}.Select(item=>item.GetProperty("file").GetString()!).ToArray();
    video=Read(inputFiles[0]); audio=Read(inputFiles[1]); text=Read(inputFiles[2]);
}
var shape=new MiniMaxH3Shape {Width=latentW*16,Height=latentH*16,
    Frames=MiniMaxH3Geometry.LatentFramesToVideoFrames(latentT),Fps=24,
    LatentWidth=latentW,LatentHeight=latentH,LatentFrames=latentT,AudioLatentFrames=audioCount/2};
void Write(string name,float[] values)
{
    if(values.Any(value=>!float.IsFinite(value))) throw new InvalidDataException("Nonfinite output");
    using var writer=new BinaryWriter(File.Create(Path.Combine(trace,name+".f32")));
    foreach(float value in values)writer.Write(value);
}
double diagnosticSeconds;
var forwardSeconds=new List<double>();
string? velocitySha256=null;
int blocks;
using(var allocator=backend == BackendType.Cuda ? new CudaAllocator(0) : null)
using(var dit=new MiniMaxH3Denoiser(model,backend,allocator))
{
    if(dit.Config.NumLayers!=50 || text.LongLength!=(long)textCount*dit.Config.TextDim)
        throw new InvalidDataException("Full50 checkpoint and matching text feature width are required");
    var layout=MiniMaxH3Layout.Build(textCount,shape,sigma);
    for(int iteration=0;iteration<warmup+repeats;iteration++)
    {
        var timer=Stopwatch.StartNew();
        var(videoOut,aVelocity)=dit.Forward(video,shape.VideoTokenCount,audio,shape.AudioTokenCount,text,textCount,layout,sigma);
        float[] vVelocity=(float[])videoOut.Clone();   // the step's output buffer, rewritten by the next step
        timer.Stop();
        forwardSeconds.Add(timer.Elapsed.TotalSeconds);
        MiniMaxH3Scheduler.ConvertAudioVelocityToCarrier(audio,aVelocity,sigma);
        byte[] values=new byte[(vVelocity.Length+aVelocity.Length)*sizeof(float)];
        Buffer.BlockCopy(vVelocity,0,values,0,vVelocity.Length*sizeof(float));
        Buffer.BlockCopy(aVelocity,0,values,vVelocity.Length*sizeof(float),aVelocity.Length*sizeof(float));
        string current=Convert.ToHexStringLower(SHA256.HashData(values));
        if(velocitySha256!=null && velocitySha256!=current) throw new InvalidDataException("Repeated identical forward changed velocity");
        velocitySha256=current;
        if(iteration==warmup+repeats-1) { Write("carrier-audio-velocity",aVelocity); Write("video-velocity",vVelocity); }
    }
    diagnosticSeconds=forwardSeconds[0];
    blocks=dit.Config.NumLayers;
}
string Sha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
var inputs=inputFiles.Select(file=>new{file,sha256=Sha(Path.Combine(inputRoot,file))});
var result=new {completed=true,backend=backendName,sigma,geometry=new[]{shape.Width,shape.Height,shape.Frames},
    textCount,audioCount,videoCount=shape.VideoTokenCount,blocks,inputs,inputMode,
    bf16Pedantic=Environment.GetEnvironmentVariable("TS_H3_CUDA_BF16_PEDANTIC") != "0",trace,
    quantBands=Environment.GetEnvironmentVariable("TS_H3_CUDA_F32_BANDS") != "0",
    quantBandMode=Environment.GetEnvironmentVariable("TS_H3_CUDA_F32_BANDS") ?? "pedantic (default)",traceEnabled,warmup,repeats,forwardSeconds,
    diagnosticSeconds,velocitySha256,
    timingScope=traceEnabled ? "Synchronized diagnostic boundaries, excluded from performance" : "Full50 Forward including host outputs; first includes device weight upload, later same-model forwards; OS page cache not controlled",
    limitation=(fixtures==null ? "Exact SD callback projection inputs" : "Synthetic fixture inputs at sigma one; original two-block fixture outputs are not reused")+
        "; excludes text encoding, sampler loop, VAE and output codec"};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output,JsonSerializer.Serialize(result,new JsonSerializerOptions{WriteIndented=true})+"\n");
Console.WriteLine(JsonSerializer.Serialize(result));
return 0;
