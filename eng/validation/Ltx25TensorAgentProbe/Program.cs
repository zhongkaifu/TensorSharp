using System.Diagnostics;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text.Json;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Shell;
using TensorSharp.Models.Ltx25;
using TensorSharp.Models.Media;
using TensorSharp.Server;

string Arg(string key,string fallback="") {int i=Array.IndexOf(args,key);return i>=0?args[i+1]:fallback;}
string assets=Path.GetFullPath(Arg("--assets")),backend=Arg("--backend","mlx");
string workspace=Path.GetFullPath(Arg("--workspace",Path.Combine("artifacts","ltx25","tensoragent-e2e-"+Guid.NewGuid().ToString("N"))));
string output=Path.GetFullPath(Arg("--output",Path.Combine(workspace,"report.json")));
string mode=Arg("--mode");bool identityMode=mode=="ref";
int width=int.Parse(Arg("--width",identityMode?"768":"512")),height=int.Parse(Arg("--height",identityMode?"448":"512")),frames=int.Parse(Arg("--frames",identityMode?"121":"17")),fps=int.Parse(Arg("--fps","24"));
long seed=long.Parse(Arg("--seed","42"));string prompt=Arg("--prompt","A small red fox walks through a quiet grassy meadow, natural daylight, gentle camera motion.");
bool verify=bool.Parse(Arg("--verify-hashes","true"));
int repeats=int.Parse(Arg("--repeats","1"));
bool chatTurn=bool.Parse(Arg("--chat-turn","true"));
string firstImage=Arg("--image"),endImage=Arg("--end-image");
string referenceImagesFile=Arg("--reference-images");
string[] referenceImages=string.IsNullOrWhiteSpace(referenceImagesFile)?[]:
    JsonSerializer.Deserialize<string[]>(File.ReadAllText(referenceImagesFile))??throw new InvalidDataException("Expected an image-path JSON array.");
if(referenceImages.Length>0&&(!string.IsNullOrWhiteSpace(firstImage)||!string.IsNullOrWhiteSpace(endImage)))
    throw new ArgumentException("Use ordered images or first/last aliases.");
if(repeats<0||(repeats==0&&!chatTurn))throw new ArgumentOutOfRangeException("--repeats","At least one HTTP generation or normal chat is required.");
if(File.Exists(output))throw new IOException("Refusing to overwrite an existing evidence report: "+output);
string stage="catalog and pinned asset preflight";
var executions=new List<object>();
object? chatExecution=null;
try
{
CatalogModel entry=ModelCatalog.Find("ltx-2.5-distilled-q4km")??throw new InvalidOperationException("Missing LTX catalog entry.");
if(!entry.IsAvailable)throw new NotSupportedException(entry.UnavailableReason);
if(backend is not ("mlx" or "cuda"))throw new ArgumentException("This real generation probe requires mlx or cuda.");
Directory.CreateDirectory(workspace);
var paths=new AgentPaths(Path.Combine(workspace,"data"),Path.Combine(workspace,"cache")){DeviceMemoryGB=48,DeviceClass=DeviceClass.Desktop,ExecutionMode=AgentExecutionMode.InProcess};
string destination=Path.Combine(paths.ModelsDirectory,entry.Id);Directory.CreateDirectory(destination);
var installed=new List<object>();
foreach(CatalogFile file in entry.Files)
{
    string subdirectory=file.Role switch{CatalogFileRole.TextEncoder=>"text_encoders",CatalogFileRole.Vae or CatalogFileRole.AudioVae=>"vae",_=>""};
    string source=Path.Combine(assets,subdirectory,file.FileName);
    using(var stream=File.OpenRead(source))
    {
        if(stream.Length!=file.Bytes)throw new InvalidDataException("Incorrect asset size: "+source);
        if(verify&&Convert.ToHexStringLower(SHA256.HashData(stream))!=file.Sha256)throw new InvalidDataException("Incorrect asset SHA256: "+source);
    }
    string link=Path.Combine(destination,file.FileName);
    if(File.Exists(link))
    {
        string? target=new FileInfo(link).ResolveLinkTarget(true)?.FullName;
        string expectedTarget=new FileInfo(source).ResolveLinkTarget(true)?.FullName??source;
        if(target!=expectedTarget)throw new IOException("Refusing to replace an existing cache entry: "+link);
    }
    else File.CreateSymbolicLink(link,source);
    installed.Add(new{file.FileName,source,link,file.Bytes,expected_sha256=file.Sha256,sha256_verified=verify});
}
if(identityMode)
{
    const string name="ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors";
    const string expected="ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95";
    string source=Path.Combine(assets,"loras",name),link=Path.Combine(destination,"loras",name);
    if(!File.Exists(source)||new FileInfo(source).Length!=1308787472||FileHash(source)!=expected)
        throw new InvalidDataException("The approved Ingredients adapter is unavailable or differs from the pinned bytes.");
    Directory.CreateDirectory(Path.GetDirectoryName(link)!);
    if(!File.Exists(link))File.CreateSymbolicLink(link,source);
    else if((new FileInfo(link).ResolveLinkTarget(true)?.FullName??link)!=(new FileInfo(source).ResolveLinkTarget(true)?.FullName??source))
        throw new IOException("Refusing to replace an existing identity adapter.");
    installed.Add(new{fileName=name,source,link,bytes=new FileInfo(source).Length,expected_sha256=expected,sha256_verified=true});
}
using var host=new AgentAppHost(paths,backends:[new BackendOption(backend,backend)]);
if(host.Models.StateOf(entry)!=InstallState.Installed)throw new InvalidOperationException("The real ModelStore does not consider the symlinked pinned assets installed.");
host.Settings.Update(s=>{s.AllowCodeExecution=false;s.AllowNetwork=false;s.SpeculativeDecoding=false;return s;});
stage="native model activation";
var load=Stopwatch.StartNew();string selected=host.UseModel(entry,warmAfterwards:false);load.Stop();
if(host.ModelService.Model is not Ltx25Model||selected!=backend)throw new InvalidOperationException("AgentAppHost did not activate the native LTX model on the requested backend.");
host.Server.Start();
using var client=new HttpClient{BaseAddress=new Uri(host.Server.BaseUrl),Timeout=TimeSpan.FromMinutes(30)};
client.DefaultRequestHeaders.Add("Cookie",$"{LoopbackServer.TokenCookie}={host.Server.Token}");
stage="reference image upload";
var uploads=new List<object>();
async Task<string> UploadImage(string path)
{
    string absolute=Path.GetFullPath(path);
    using var form=new MultipartFormDataContent();
    using var bytes=new ByteArrayContent(await File.ReadAllBytesAsync(absolute));
    bytes.Headers.ContentType=new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
    form.Add(bytes,"file",Path.GetFileName(absolute));
    using var response=await client.PostAsync("/api/upload",form);
    string payload=await response.Content.ReadAsStringAsync();
    response.EnsureSuccessStatusCode();
    using var document=JsonDocument.Parse(payload);
    JsonElement reply=document.RootElement;
    if(reply.GetProperty("mediaType").GetString()!="image")throw new InvalidDataException("Uploaded guide was not admitted as an image.");
    string stored=reply.GetProperty("file").GetString()!;
    byte[] served=await client.GetByteArrayAsync(reply.GetProperty("url").GetString()!);
    if(Convert.ToHexStringLower(SHA256.HashData(served))!=FileHash(absolute))throw new InvalidDataException("Uploaded image changed when downloaded.");
    uploads.Add(new{path=absolute,sha256=FileHash(absolute),bytes=new FileInfo(absolute).Length,stored,reply=reply.Clone()});
    return stored;
}
string? uploadedFirst=string.IsNullOrWhiteSpace(firstImage)?null:await UploadImage(firstImage);
string? uploadedEnd=string.IsNullOrWhiteSpace(endImage)?null:await UploadImage(endImage);
var uploadedReferences=new List<string>();
foreach(string image in referenceImages)uploadedReferences.Add(await UploadImage(image));
var body=new Dictionary<string,object>{["prompt"]=prompt,["width"]=width,["height"]=height,["frames"]=frames,["fps"]=fps,
    ["steps"]=8,["cfg"]=1,["sampler"]="euler_ancestral",["flowShift"]=0,["seed"]=seed,["videoMode"]="t2va",["generateAudio"]=true};
if(uploadedFirst!=null)body["imagePath"]=uploadedFirst;
if(uploadedEnd!=null)body["endImage"]=uploadedEnd;
if(uploadedReferences.Count>0){body["referenceImages"]=uploadedReferences;body["videoMode"]="keyframes";}
else if(uploadedFirst!=null||uploadedEnd!=null)body["videoMode"]=uploadedEnd!=null?"fl2v":"i2v"; // fl2v: a last frame, optionally with a first
if(!string.IsNullOrWhiteSpace(mode))body["videoMode"]=mode;
for(int repeat=0;repeat<repeats;++repeat)
{
long cacheHitsBefore=((Ltx25Model)host.ModelService.Model).PromptCacheHits;
stage="native generation through HTTP, repeat "+repeat;
var events=new List<JsonElement>();JsonElement terminal=default;var elapsed=Stopwatch.StartNew();
using(var request=new HttpRequestMessage(HttpMethod.Post,"/api/video-generate/stream"){Content=JsonContent.Create(body)})
using(var response=await client.SendAsync(request,HttpCompletionOption.ResponseHeadersRead))
{
    response.EnsureSuccessStatusCode();
    if(response.Content.Headers.ContentType?.MediaType!="text/event-stream")throw new InvalidDataException("Video job did not use the real SSE transport.");
    using var reader=new StreamReader(await response.Content.ReadAsStreamAsync());
    while(await reader.ReadLineAsync() is { } line)
    {
        if(!line.StartsWith("data: ",StringComparison.Ordinal))continue;
        using var item=JsonDocument.Parse(line[6..]);var e=item.RootElement.Clone();events.Add(e);
        if(e.TryGetProperty("error",out var error))throw new InvalidOperationException(error.ToString());
        if(e.TryGetProperty("done",out var done)&&done.ValueKind==JsonValueKind.True)terminal=e;
    }
}
elapsed.Stop();
if(terminal.ValueKind!=JsonValueKind.Object)throw new InvalidDataException("Missing completed video job.");
foreach(var property in new[]{("width",width),("height",height),("frames",frames),("fps",fps)})
    if(terminal.GetProperty(property.Item1).GetInt32()!=property.Item2)throw new InvalidDataException("Incorrect output "+property.Item1);
if(terminal.GetProperty("seed").GetInt64()!=seed)throw new InvalidDataException("Incorrect output seed.");
if(terminal.GetProperty("codec").GetString()!="h264"||!terminal.GetProperty("audioMuxed").GetBoolean())throw new InvalidDataException("The playable MP4 must contain H.264 video and its generated soundtrack.");
stage="HTTP media download and validation";
string video=Path.Combine(workspace,$"tensoragent-ltx25-{repeat}.mp4"),audio=Path.Combine(workspace,$"tensoragent-ltx25-{repeat}.wav");
await File.WriteAllBytesAsync(video,await client.GetByteArrayAsync(terminal.GetProperty("url").GetString()!));
await File.WriteAllBytesAsync(audio,await client.GetByteArrayAsync(terminal.GetProperty("audioUrl").GetString()!));
byte[] wav=await File.ReadAllBytesAsync(audio);
int audioTokens=Math.Max(1,(int)Math.Round(frames/(double)fps*25,MidpointRounding.ToEven));
int expectedSamples=(4*audioTokens-3)*480;
if(wav.Length!=44+expectedSamples*4||BitConverter.ToInt32(wav,24)!=48000||BitConverter.ToInt16(wav,22)!=2)throw new InvalidDataException("Audio sidecar is not 48 kHz stereo WAV.");
double nativeMs=events.Where(e=>e.TryGetProperty("phase",out var p)&&p.GetString()=="done")
    .Select(e=>e.GetProperty("elapsedSeconds").GetDouble()*1000).DefaultIfEmpty(double.NaN).Last();
if(!double.IsFinite(nativeMs))throw new InvalidDataException("Missing authentic native generation completion timing.");
executions.Add(new{repeat,conditioning_cache_hit=((Ltx25Model)host.ModelService.Model).PromptCacheHits>cacheHitsBefore,
    native_generation_ms=nativeMs,whole_http_job_ms=elapsed.Elapsed.TotalMilliseconds,terminal,events,video,audio,
    video_sha256=FileHash(video),audio_sha256=FileHash(audio),video_bytes=new FileInfo(video).Length,audio_bytes=wav.Length,audio_samples=expectedSamples});
}
if(chatTurn)
{
    stage="normal user chat session";
    using var sessionResponse=await client.PostAsJsonAsync("/api/sessions",new{});
    sessionResponse.EnsureSuccessStatusCode();
    using var session=JsonDocument.Parse(await sessionResponse.Content.ReadAsStringAsync());
    string sessionId=session.RootElement.GetProperty("sessionId").GetString()!;
    string conversationId=session.RootElement.GetProperty("conversationId").GetString()!;
    string[] chatImages=uploadedReferences.Count>0?uploadedReferences.ToArray():new[]{uploadedFirst,uploadedEnd}.Where(path=>path!=null).Select(path=>path!).ToArray();
    var chatUser=new Dictionary<string,object>{["role"]="user",["content"]=prompt};
    if(chatImages.Length>0)
    {
        chatUser["stillImagePaths"]=chatImages;
        chatUser["imagePaths"]=chatImages;
        chatUser["attachments"]=chatImages.Select(path=>new{file=path,fileName=path,mediaType="image"}).ToArray();
    }
    if(!string.IsNullOrWhiteSpace(mode))chatUser["videoMode"]=mode;
    if(uploadedFirst==null&&uploadedEnd!=null)
        chatUser["imageConditions"]=new[]{new{imagePath=uploadedEnd,frameIndex=-1,strength=1}};
    var chatBody=new{sessionId,messages=new[]{chatUser}};
    long cacheHitsBefore=((Ltx25Model)host.ModelService.Model).PromptCacheHits;
    var chatElapsed=Stopwatch.StartNew();
    var chatEvents=await ReadSse(client,"/api/chat",chatBody);
    chatElapsed.Stop();
    JsonElement videoFrame=chatEvents.LastOrDefault(e=>e.TryGetProperty("videoUrl",out var url)&&url.ValueKind==JsonValueKind.String);
    if(videoFrame.ValueKind!=JsonValueKind.Object||!chatEvents.Any(e=>e.TryGetProperty("done",out var done)&&done.ValueKind==JsonValueKind.True))
        throw new InvalidDataException("Normal chat did not finish with a generated video.");
    foreach(var expected in new[]{("width",identityMode?768:512),("height",identityMode?448:512),("frames",identityMode?121:17),("fps",24)})
        if(videoFrame.GetProperty(expected.Item1).GetInt32()!=expected.Item2)throw new InvalidDataException("The normal chat did not apply the LTX recipe: "+expected.Item1);
    if(!videoFrame.GetProperty("hasAudio").GetBoolean()||videoFrame.GetProperty("audioUrl").ValueKind!=JsonValueKind.Null)
        throw new InvalidDataException("Normal chat must deliver its generated soundtrack inside the MP4.");
    if(repeats>0&&((Ltx25Model)host.ModelService.Model).PromptCacheHits<=cacheHitsBefore)
        throw new InvalidDataException("The normal chat did not reuse the same model's prompt conditioning.");
    long actualSeed=videoFrame.GetProperty("seed").GetInt64();
    string videoUrl=videoFrame.GetProperty("videoUrl").GetString()!;
    string audioUrl=Path.ChangeExtension(videoUrl,".wav");
    string chatVideo=Path.Combine(workspace,"tensoragent-chat.mp4"),chatAudio=Path.Combine(workspace,"tensoragent-chat.wav");
    await File.WriteAllBytesAsync(chatVideo,await client.GetByteArrayAsync(videoUrl));
    await File.WriteAllBytesAsync(chatAudio,await client.GetByteArrayAsync(audioUrl));
    byte[] chatWav=await File.ReadAllBytesAsync(chatAudio);
    int chatFrames=identityMode?121:17;
    int chatSamples=(4*(int)Math.Round(chatFrames/24d*25,MidpointRounding.ToEven)-3)*480;
    if(chatWav.Length!=44+chatSamples*4||BitConverter.ToInt32(chatWav,24)!=48000||BitConverter.ToInt16(chatWav,22)!=2)
        throw new InvalidDataException("The normal chat soundtrack has incorrect native geometry.");
    using var saved=JsonDocument.Parse(await client.GetStringAsync("/api/agent/conversations/"+conversationId));
    JsonElement conversation=saved.RootElement.Clone();
    var messages=conversation.GetProperty("messages").EnumerateArray().ToArray();
    if(!messages.Any(m=>m.GetProperty("role").GetString()=="user"&&m.GetProperty("content").GetString()==prompt)
        ||!messages.Any(m=>m.GetProperty("role").GetString()=="assistant"&&m.TryGetProperty("videoUrl",out var url)&&url.GetString()==videoUrl))
        throw new InvalidDataException("The completed chat and generated clip were not persisted in the conversation.");
    chatExecution=new{status="passed",scope="Normal choose-and-chat route: real session binding, recipe defaults, managed turn, conversation recorder and media download; random seed is descriptive, not a matched benchmark.",
        request=chatBody,sessionId,conversationId,seed=actualSeed,conditioning_cache_hit=((Ltx25Model)host.ModelService.Model).PromptCacheHits>cacheHitsBefore,whole_http_chat_ms=chatElapsed.Elapsed.TotalMilliseconds,
        video_frame=videoFrame,events=chatEvents,conversation,video=chatVideo,audio=chatAudio,video_sha256=FileHash(chatVideo),audio_sha256=FileHash(chatAudio),audio_samples=chatSamples};
}
var report=new{status="passed",provenance=Provenance(),entry.Id,backend=selected,model_type=host.ModelService.Model.GetType().FullName,
    scope="Real AgentAppHost.UseModel, pinned cache assets, native full generation and WebUI SSE/upload download pipeline",
    timing_scope="Native generation comes from the model's final progress callback; HTTP time also includes codec/export and transport. Model activation is separate; pinned setup hashing is excluded. File and JIT caches may be warm.",
    installed,uploads,request=body,load_ms=load.Elapsed.TotalMilliseconds,executions,chat_execution=chatExecution};
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
await File.WriteAllTextAsync(output,JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true})+"\n");Console.WriteLine(output);

}
catch(Exception error)
{
    Directory.CreateDirectory(Path.GetDirectoryName(output)!);
    File.WriteAllText(output,JsonSerializer.Serialize(new{status="failed",stage,backend,workspace,executions,chat_execution=chatExecution,error_type=error.GetType().FullName,error=error.Message,provenance=Provenance()},new JsonSerializerOptions{WriteIndented=true})+"\n");
    throw;
}
static object Provenance()
{
    var managed=AppDomain.CurrentDomain.GetAssemblies().Where(a=>!a.IsDynamic&&(a.GetName().Name?.StartsWith("TensorSharp",StringComparison.Ordinal)==true||a.GetName().Name?.StartsWith("TensorAgent",StringComparison.Ordinal)==true))
        .Where(a=>File.Exists(a.Location)).ToDictionary(a=>a.GetName().Name!,a=>new{path=a.Location,sha256=FileHash(a.Location)});
    string? ffmpeg=Environment.GetEnvironmentVariable("TS_FFMPEG");
    if(!File.Exists(ffmpeg))ffmpeg=(Environment.GetEnvironmentVariable("PATH")??"").Split(Path.PathSeparator).Select(p=>Path.Combine(p,OperatingSystem.IsWindows()?"ffmpeg.exe":"ffmpeg")).FirstOrDefault(File.Exists);
    var policy=new[]{"MLX_ENABLE_TF32","TS_MLX_CACHE_LIMIT_MB","TS_MLX_WIRED_LIMIT_MB","TS_MLX_MEMORY_LIMIT_MB",
        "TS_LTX25_MLX_BF16_GEMM","TS_LTX25_CUDA_BF16_GEMM","TS_LTX25_MLX_HALF_PROJECTION","TS_LTX25_CUDA_ATTN","TS_LTX25_CUDA_TILE"}
        .ToDictionary(name=>name,Environment.GetEnvironmentVariable);
    var modules=new[]{"tensorsharp_kernels.ptx","tensorsharp_ltx25.ptx"}.Select(name=>Path.Combine(AppContext.BaseDirectory,"cuda_kernels",name))
        .Where(File.Exists).ToDictionary(path=>Path.GetFileName(path),path=>new{path,sha256=FileHash(path)});
    return new{managed,policy,modules,ffmpeg,ffmpeg_sha256=ffmpeg is null?null:FileHash(ffmpeg),image_codec=MediaCodecs.Image.GetType().FullName,video_encoder=MediaCodecs.VideoEncoder.GetType().FullName};
}
static string FileHash(string path){using var stream=File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(stream));}

static async Task<List<JsonElement>> ReadSse(HttpClient client,string path,object body)
{
    using var request=new HttpRequestMessage(HttpMethod.Post,path){Content=JsonContent.Create(body)};
    using var response=await client.SendAsync(request,HttpCompletionOption.ResponseHeadersRead);
    response.EnsureSuccessStatusCode();
    if(response.Content.Headers.ContentType?.MediaType!="text/event-stream")throw new InvalidDataException("The normal chat did not use SSE.");
    var events=new List<JsonElement>();
    using var reader=new StreamReader(await response.Content.ReadAsStreamAsync());
    while(await reader.ReadLineAsync() is { } line)
    {
        if(!line.StartsWith("data: ",StringComparison.Ordinal))continue;
        using var item=JsonDocument.Parse(line[6..]);var value=item.RootElement.Clone();events.Add(value);
        if(value.TryGetProperty("error",out var error))throw new InvalidOperationException(error.ToString());
        if(value.TryGetProperty("aborted",out var aborted)&&aborted.ValueKind==JsonValueKind.True)throw new InvalidOperationException("The normal chat was aborted.");
    }
    return events;
}
