using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

if (args.Length != 6) throw new ArgumentException("model head plain.rows plain.inputs mtp.rows output");
string output=args[5];if(Directory.Exists(output))throw new IOException("Fresh output required");Directory.CreateDirectory(output);
using var rows=JsonDocument.Parse(File.ReadAllBytes(args[2]));using var inputs=JsonDocument.Parse(File.ReadAllBytes(args[3]));using var mtp=JsonDocument.Parse(File.ReadAllBytes(args[4]));
using var raw=ModelBase.Create(args[0],BackendType.GgmlCuda,tpDegree:3,draftModelPath:args[1]);
var model=raw as Qwen4ExpModel??throw new InvalidOperationException("Qwen target required");
if(!model.HasDraftHead||!model.SpeculationProfitable)throw new InvalidOperationException("Attached head and target path required");
int vocab=model.Config.VocabSize;var reports=new List<object>();
foreach(var label in new[]{"json","retained-A"})
{
 var row=rows.RootElement.EnumerateArray().Single(x=>x.GetProperty("Label").GetString()==label);
 var mr=mtp.RootElement.EnumerateArray().Single(x=>x.GetProperty("Label").GetString()==label);
 int[] tokens=row.GetProperty("Tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 int[] other=mr.GetProperty("Tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 int mismatch=Enumerable.Range(0,Math.Min(tokens.Length,other.Length)).First(i=>tokens[i]!=other[i]);
 var input=inputs.RootElement.GetProperty("inputs").EnumerateArray().Single(x=>x.GetProperty("id").GetString()==label);
 int[] prompt=input.GetProperty("tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 model.ResetKVCache();model.ForwardRefill(prompt);var reference=new float[mismatch][];
 using(var file=File.Create(Path.Combine(output,label+"-scalar.f32")))
  for(int i=0;i<mismatch;i++){reference[i]=(float[])model.Forward([tokens[i]]).Clone();WriteFloats(file,reference[i]);}
 foreach(int width in new[]{1,2,3,4})
 {
  model.ResetKVCache();model.ForwardRefill(prompt);var records=new List<object>();int unequal=0,argmaxChanges=0;double peak=0;
  using var vectors=File.Create(Path.Combine(output,$"{label}-blocks{width}.f32"));
  for(int start=0;start<mismatch;start+=width)
  {
   int count=Math.Min(width,mismatch-start);var actual=new float[count*vocab];var hidden=new float[count*model.SpecFeatureSize];
   model.SpecForward(tokens[start..(start+count)],hidden,actual,allLogitsRows:true);WriteFloats(vectors,actual);
   for(int r=0;r<count;r++)
   {
    var got=actual.AsSpan(r*vocab,vocab);var expected=reference[start+r];double max=0,sq=0,den=0;int changed=0;
    for(int j=0;j<vocab;j++){if(!float.IsFinite(got[j])||!float.IsFinite(expected[j]))throw new ArithmeticException("Nonfinite full vocabulary");double d=(double)got[j]-expected[j];max=Math.Max(max,Math.Abs(d));sq+=d*d;den+=(double)expected[j]*expected[j];if(BitConverter.SingleToInt32Bits(got[j])!=BitConverter.SingleToInt32Bits(expected[j]))changed++;}
    int pa=Argmax(expected),va=Argmax(got);if(changed>0)unequal++;if(pa!=va)argmaxChanges++;peak=Math.Max(peak,max);
    records.Add(new{token_position=start+r,block_start=start,block_count=count,changed,max_abs=max,relative_l2=Math.Sqrt(sq/Math.Max(den,1e-30)),plain_argmax=pa,verify_argmax=va,original_next_token=tokens[start+r+1],original_mtp_next_token=other[start+r+1],plain_score=expected[tokens[start+r+1]],verify_score=got[tokens[start+r+1]],alternative_plain_score=expected[other[start+r+1]],alternative_verify_score=got[other[start+r+1]]});
   }
  }
  reports.Add(new{label,width,vocab,first_original_output_difference=mismatch,prompt_tokens=prompt.Length,prompt_sha256=input.GetProperty("tokens_i32_sha256").GetString(),unequal_rows=unequal,argmax_changes=argmaxChanges,max_abs=peak,rows=records});
  File.WriteAllText(Path.Combine(output,"result.json"),JsonSerializer.Serialize(new{release_qualified=false,performance_qualified=false,scope="Identical full common generated history through successive committed target blocks, versus scalar teacher forcing. Full-vocabulary row-major F32 files. Diagnostic only: no grammar, draft proposals or scheduler rollback.",cases=reports},new JsonSerializerOptions{WriteIndented=true}));
  Console.WriteLine($"{label} width={width} unequal={unequal}/{mismatch} argmax_changes={argmaxChanges} max_abs={peak:G9}");
 }
}
static int Argmax(ReadOnlySpan<float> a){int best=0;for(int i=1;i<a.Length;i++)if(a[i]>a[best])best=i;return best;}
static void WriteFloats(Stream stream,float[] a){stream.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan()));}
