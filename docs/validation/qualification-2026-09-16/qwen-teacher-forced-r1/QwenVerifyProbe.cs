using System.Security.Cryptography;
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
var reports=new List<object>();
foreach(var label in new[]{"json","retained-A"})
{
 var row=rows.RootElement.EnumerateArray().Single(x=>x.GetProperty("Label").GetString()==label);
 var mr=mtp.RootElement.EnumerateArray().Single(x=>x.GetProperty("Label").GetString()==label);
 int[] tokens=row.GetProperty("Tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 int[] other=mr.GetProperty("Tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 int mismatch=Enumerable.Range(0,Math.Min(tokens.Length,other.Length)).First(i=>tokens[i]!=other[i]);
 var input=inputs.RootElement.GetProperty("inputs").EnumerateArray().Single(x=>x.GetProperty("id").GetString()==label);
 int[] prompt=input.GetProperty("tokens").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
 for(int width=1;width<=4;width++)
 {
  int start=mismatch-width;if(start<0)throw new InvalidOperationException();
  model.ResetKVCache();model.ForwardRefill(prompt);
  for(int i=0;i<start;i++)model.Forward([tokens[i]]);
  var reference=new float[width][];
  for(int i=0;i<width;i++)reference[i]=(float[])model.Forward([tokens[start+i]]).Clone();
  model.ResetKVCache();model.ForwardRefill(prompt);
  for(int i=0;i<start;i++)model.Forward([tokens[i]]);
  var actual=new float[width*model.Config.VocabSize];var hidden=new float[width*model.SpecFeatureSize];
  model.SpecForward(tokens[start..mismatch],hidden,actual,allLogitsRows:true);
  var records=new List<object>();
  for(int i=0;i<width;i++)
  {
   var got=actual.AsSpan(i*model.Config.VocabSize,model.Config.VocabSize).ToArray();var expected=reference[i];
   if(got.Any(x=>!float.IsFinite(x))||expected.Any(x=>!float.IsFinite(x)))throw new ArithmeticException("Nonfinite full vocabulary");
   double max=0,sq=0,den=0;int changed=0;
   for(int j=0;j<got.Length;j++){double d=(double)got[j]-expected[j];max=Math.Max(max,Math.Abs(d));sq+=d*d;den+=(double)expected[j]*expected[j];if(BitConverter.SingleToInt32Bits(got[j])!=BitConverter.SingleToInt32Bits(expected[j]))changed++;}
   string stem=$"{label}-w{width}-row{i}";
   Save(stem+"-plain.f32",expected);Save(stem+"-verify.f32",got);
   records.Add(new{row=i,token_position=start+i,changed,max_abs=max,relative_l2=Math.Sqrt(sq/Math.Max(den,1e-30)),plain_argmax=Argmax(expected),verify_argmax=Argmax(got),original_next_token=tokens[start+i+1],original_mtp_next_token=other[start+i+1],plain_score=expected[tokens[start+i+1]],verify_score=got[tokens[start+i+1]],alternative_plain_score=expected[other[start+i+1]],alternative_verify_score=got[other[start+i+1]]});
   Console.WriteLine($"{stem} changed={changed} max={max:G9} argmax={Argmax(expected)}/{Argmax(got)}");
  }
  reports.Add(new{label,width,first_original_output_difference=mismatch,prompt_tokens=prompt.Length,prompt_sha256=input.GetProperty("tokens_i32_sha256").GetString(),rows=records});
  File.WriteAllText(Path.Combine(output,"result.json"),JsonSerializer.Serialize(new{release_qualified=false,performance_qualified=false,scope="Identical teacher-forced tokens and full-vocabulary logits; block verification versus scalar target decode. Diagnostic only, no grammar or scheduler replay.",cases=reports},new JsonSerializerOptions{WriteIndented=true}));
 }
}
static int Argmax(float[] a){int best=0;for(int i=1;i<a.Length;i++)if(a[i]>a[best])best=i;return best;}
void Save(string name,float[] a){byte[] b=new byte[a.Length*4];Buffer.BlockCopy(a,0,b,0,b.Length);File.WriteAllBytes(Path.Combine(output,name),b);}
