using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;

using var json = JsonDocument.Parse(File.ReadAllText("/workspace/ts-nemotron-audio-research/trained_reference.json"));
var root = json.RootElement;
string path = "/workspace/models/nemotron-omni/audio-e5e9932/mmproj-audio-bf16.gguf";
using(var file = File.OpenRead(path))
    if (Convert.ToHexStringLower(SHA256.HashData(file)) != root.GetProperty("gguf_sha256").GetString()) throw new Exception("Companion identity mismatch");
GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cpu);
var allocator = new GgmlAllocator(new GgmlContext(new[] {0},GgmlBackendType.Cpu),0);
var watch=Stopwatch.StartNew();
using var details=JsonDocument.Parse(File.ReadAllText("/workspace/ts-nemotron-audio-research/trained_details.json"));
var diffs=new List<object>();
Action<string,float[]> trace=(name,values)=>{if(!details.RootElement.TryGetProperty(name,out var entry))return;float[] target=entry.EnumerateArray().Select(v=>v.GetSingle()).ToArray();if(values.Length!=target.Length)return;double maxdiff=values.Zip(target,(a,b)=>(double)Math.Abs(a-b)).Max();int count=values.Zip(target,(a,b)=>a!=b?1:0).Sum();if(count>0){diffs.Add(new{name,maxdiff,count,total=values.Length,actual=values,expected=target});Console.WriteLine($"stage={name} maxdiff={maxdiff:R} changed={count}/{values.Length}");}};
var constructor=typeof(NemotronAudioEncoder).GetConstructor(System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic,null,new[]{typeof(string),typeof(IAllocator),typeof(Action<string,float[]>)},null)!;
using var encoder=(NemotronAudioEncoder)constructor.Invoke(new object[]{path,allocator,trace});
double load=watch.Elapsed.TotalSeconds;
float[] input=root.GetProperty("input").EnumerateArray().Select(v=>v.GetSingle()).ToArray();
float[] expected=root.GetProperty("expected").EnumerateArray().Select(v=>v.GetSingle()).ToArray();
watch.Restart();
using var result=encoder.Encode(input,root.GetProperty("frames").GetInt32(),root.GetProperty("valid_frames").GetInt32());
float[] actual=result.GetElementsAsFloat(expected.Length);
double encode=watch.Elapsed.TotalSeconds;
int failed=0,worstIndex=0;double max=0,sum2=0,expected2=0;
for(int i=0;i<actual.Length;i++)
{
    double delta=Math.Abs(actual[i]-expected[i]);
    if(!double.IsFinite(delta)||delta>0.001+0.01*Math.Abs(expected[i]))failed++;
    if(delta>max){max=delta;worstIndex=i;}sum2+=delta*delta;expected2+=expected[i]*expected[i];
}
var report=new {shape=result.Sizes.ToArray(),load_seconds=load,encode_seconds=encode,failed_elements=failed,total_elements=actual.Length,max_abs=max,relative_rmse=Math.Sqrt(sum2/expected2),worst_index=worstIndex,worst_expected=expected[worstIndex],worst_actual=actual[worstIndex],peak_working_set=Process.GetCurrentProcess().PeakWorkingSet64,actual};
string output=JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true});
File.WriteAllText("trained-result.json",output);
File.WriteAllText("trained-diffs.json",JsonSerializer.Serialize(diffs));
Console.WriteLine(JsonSerializer.Serialize(new{failed,max,relative_rmse=Math.Sqrt(sum2/expected2),load,encode,peak_working_set=Process.GetCurrentProcess().PeakWorkingSet64}));
return failed==0?0:1;
