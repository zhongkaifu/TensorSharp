// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Validation;
using static Support;

internal static class Program
{
    public static int Main(string[] args)
    {
        string? output = null; bool owned = false;
        var report = new Dictionary<string,object?> { ["schema"]="ninfer-residual-norm-body/1",["complete"]=false,["passed"]=false,["GPU"]="NOT_RUN",["performance"]="NOT_MEASURED",["quality"]="NOT_RUN" };
        try
        {
            var options = Options(args);
            if (options.ContainsKey("--self-test")) { Console.WriteLine(Encoded(Protocol.SelfTest())); return 0; }
            if (options.ContainsKey("--compare")) { Compare.Run(options); return 0; }
            if (!options.ContainsKey("--execute")) { Console.WriteLine("PLAN: --execute --mode old|off|on --artifact MODEL --expect-directory-sha SHA --inputs JSON --expect-inputs-sha SHA --native-pins JSON --expect-native-pins-sha SHA --out FRESH_IGNORED_DIR. Three independent processes; fixed context40960/UB2048. --compare --old OLD_DIR --off OFF_DIR --on ON_DIR --out FRESH_IGNORED_DIR. --self-test is CPU-only."); return 0; }
            Need(BitConverter.IsLittleEndian,"Little-endian raw storage required.");
            string mode = Required(options,"--mode"); Need(mode is "old" or "off" or "on","Exact producer mode required.");
            string artifact=Path.GetFullPath(Required(options,"--artifact")), input=Path.GetFullPath(Required(options,"--inputs")), nativePath=Path.GetFullPath(Required(options,"--native-pins"));
            string directorySha=ValidationSha256.Normalize(Required(options,"--expect-directory-sha")), inputSha=ValidationSha256.Normalize(Required(options,"--expect-inputs-sha")), nativeSha=ValidationSha256.Normalize(Required(options,"--expect-native-pins-sha"));
            byte[] inputBytes=File.ReadAllBytes(input), nativeBytes=File.ReadAllBytes(nativePath);
            Need(Hash(inputBytes)==inputSha && Hash(nativeBytes)==nativeSha,"Pinned input/native manifest bytes differ.");
            var native=Protocol.Pins(nativeBytes,mode); var pool=Protocol.Inputs(inputBytes); var seeds=Protocol.Seeds(pool.Prompt,pool.Continuation);
            var env=Protocol.EnvironmentSnapshot();
            Need(Environment.GetEnvironmentVariable("TS_Q35_NINFER_VERIFY")=="1" && Environment.GetEnvironmentVariable("TS_Q35_NINFER_DECODE_GRAPHS")=="1" && Environment.GetEnvironmentVariable("TS_Q35_NINFER_VERIFY8_GEMM")=="1","Explicit original verification/graphs/BN8 controls required.");
            Need(Environment.GetEnvironmentVariable("TS_Q35_NINFER_RESIDUAL_NORM") == (mode=="on"?"1":"0"),"Candidate flag differs from declared mode.");
            var numericalEnvironment=Protocol.NumericalEnvironment(env,native);
            var artifactInfo=new FileInfo(artifact); Need(artifactInfo.Exists,"Original artifact unavailable.");
            var artifactStat=new { artifactInfo.Length,artifactInfo.LastWriteTimeUtc };
            using(var reader=new NinferArtifactReader(artifact)) Need(reader.DirectorySha256==directorySha,"Artifact directory identity differs.");
            var assemblyPins=AppDomain.CurrentDomain.GetAssemblies().Where(a=>!a.IsDynamic && !string.IsNullOrEmpty(a.Location) && Path.GetFileName(a.Location).StartsWith("TensorSharp",StringComparison.Ordinal)).Append(typeof(Program).Assembly).Distinct().Select(a=>FilePin(a.Location)).ToArray();
            var pins=new[]{FilePin(input),FilePin(nativePath)}.Concat(native.Select(p=>new Pin(p.Path,p.Bytes,p.Sha256))).Concat(assemblyPins).ToArray();
            output=Required(options,"--out");Fresh(output);Directory.CreateDirectory(output);owned=true;
            File.WriteAllBytes(Path.Combine(output,"source-inputs.json"),inputBytes);File.WriteAllBytes(Path.Combine(output,"native-pins.json"),nativeBytes);
            Save(Path.Combine(output,"derived-inputs.json"),new{seeds,largePrefill=pool.Prompt.Take(2048).ToArray(),bases=Protocol.Bases,keeps=Protocol.Keeps,originalInputSha256=inputSha});
            var store=new Store(output); var checkpoints=new List<object>();var events=new List<object>();var arguments=new List<object>();var resetReference=new Dictionary<string,string>(StringComparer.Ordinal);
            report["mode"]=mode;report["artifactDirectorySha256"]=directorySha;report["artifactStat"]=artifactStat;report["sourceInputSha256"]=inputSha;
            report["derivedInputSha256"]=FilePin(Path.Combine(output,"derived-inputs.json")).Sha256;report["context"]=Protocol.Context;report["ubatch"]=Protocol.Ubatch;
            report["nativeFiles"]=native;report["assemblies"]=assemblyPins;report["environment"]=env;report["numericalEnvironment"]=numericalEnvironment;
            report["vectors"]=store.Vectors;report["checkpoints"]=checkpoints;report["events"]=events;report["arguments"]=arguments;report["rawByteBound"]=Protocol.RawBound;report["rawByteCap"]=Budget;
            Save(Path.Combine(output,"result.json"),report); report["GPU"]="ATTEMPTED";
            int verify8=0,verify1=0,commits=0,decodes=0,proposals=0,clones=0,resets=0,largePrefills=0;
            using(var executor=new Qwen35CudaExecutor(artifact,Protocol.Context,Protocol.Ubatch))
            {
                var e=executor.Engine;report["loadedImages"]=Protocol.Loaded(e,executor,native,mode);
                report["draftPayloadSha256"]=e.NinferDraftPayloadSha256; report["targetW8"]=e.OutputW8Info;report["numericalStateNamespace"]=e.KVStateNamespaceSuffix;
                report["candidatePolicy"]=Optional(e,"NinferResidualNormPolicy");report["candidateSelection"]=Optional(e,"NinferResidualNormSelection");
                var seedSlot=e.Active;State.Counter initial=State.Counters(e);report["initialCounters"]=initial;
                void Checkpoint(string id,int pass,string phase,Q35CudaEngine.Slot slot,bool kv=true)
                {
                    var snapshot=State.Capture(e,slot,slot.NPast,kv);string key=id+"/"+phase, encoded=Encoded(State.Numeric(snapshot));
                    checkpoints.Add(new{id,pass,phase,snapshot});
                    if(pass==0)Need(resetReference.TryAdd(key,encoded),"Duplicate reset state identity.");
                    else Need(resetReference.TryGetValue(key,out string? expected)&&expected==encoded,"Reset-repeat effective state differs: "+key);
                }
                void Verify(string id,int pass,Q35CudaEngine.Slot slot,int[] tokens)
                {
                    int n=tokens.Length;var before=State.Capture(e,slot,slot.NPast,false);var ca=State.Counters(e);float[] head=Poison(n,pass);int[] drawn=Enumerable.Repeat(IntGuard,n+Guard).ToArray();
                    e.Verify(new[]{slot},n,tokens,drawn,head);e.Synchronize();var cb=State.Counters(e);State.VerifyTransition(ca,cb,mode,n);
                    events.Add(new{id,pass,operation="verify",rows=n,before=ca,after=cb});
                    store.Record(id,pass,head,n,drawn);State.Taps(e,store,id,pass,n);arguments.Add(new{id,pass,operation="verify",tokens,drawn=drawn.Take(n).ToArray()});
                    Need(Encoded(State.Numeric(before))==Encoded(State.Numeric(State.Capture(e,slot,slot.NPast,false))),"Verify mutated committed recurrent/draft state.");
                    if(n==8)verify8++;else{Need(n==1,"Only fixed1/8 verify allowed.");verify1++;}
                }
                float[] Decode(string id,int pass,Q35CudaEngine.Slot slot,int token)
                {
                    int pos=slot.NPast;float[] head=Poison(1,pass);int[] drawn=Enumerable.Repeat(IntGuard,1+Guard).ToArray();
                    e.DecodeBatch(new[]{slot},new[]{token},head,drawn);e.Synchronize();Need(slot.NPast==pos+1,"Decode position differs.");
                    store.Record(id,pass,head,1,drawn);arguments.Add(new{id,pass,operation="decode",token,drawn=drawn[0]});decodes++;return head;
                }
                void Proposal(string id,int pass,Q35CudaEngine.Slot slot,int anchor)
                {
                    int pos=slot.NPast;float[] head=Poison(7,pass);int[] ids=Enumerable.Repeat(IntGuard,7+Guard).ToArray();float[] confidence=Enumerable.Repeat(BitConverter.Int32BitsToSingle(GuardBits),7+Guard).ToArray();
                    e.ProposeNinfer(new[]{slot},new[]{anchor},new[]{ids},new[]{confidence},head);e.Synchronize();
                    Need(slot.NPast==pos && ids.Take(7).All(x=>x>=0&&x<Valid) && ids.Skip(7).All(x=>x==IntGuard)
                        && confidence.Take(7).All(x=>BitConverter.SingleToInt32Bits(x)==BitConverter.SingleToInt32Bits(1f)) && confidence.Skip(7).All(x=>BitConverter.SingleToInt32Bits(x)==GuardBits),"Proposal IDs/confidence/guards/target position differ.");
                    store.Record(id,pass,head,7);arguments.Add(new{id,pass,operation="proposal",anchor,ids=ids.Take(7).ToArray(),confidenceBits=confidence.Take(7).Select(BitConverter.SingleToInt32Bits).ToArray()});proposals++;
                    Need(State.Meta(e,slot).PendingBytes==0,"Proposal failed to flush owned pending anchor.");
                }
                Q35CudaEngine.Slot Clone(Q35CudaEngine.Slot source)
                { var slot=e.CreateSlot(0);try{e.CopySlot(source,slot);clones++;return slot;}catch{e.FreeSlot(slot);throw;} }
                void Case(Protocol.Seed seed,int keep,bool pending,int pass)
                {
                    string id=seed.Id+(pending?"/pending1":"/keep"+keep);var slot=Clone(seedSlot);e.Active=slot;
                    try
                    {
                        Checkpoint(id,pass,"initial",slot);
                        if(pending)
                        {
                            Verify(id+"/pending-anchor",pass,slot,new[]{seed.Anchor});e.Commit(new[]{slot},1,new[]{1});commits++;
                            var m=State.Meta(e,slot);Need(m.PendingBytes==51200&&m.Frontier==slot.NPast-1,"Verify1/Commit1 owned pending absent.");
                            Checkpoint(id,pass,"pending",slot);
                        }
                        int pos=slot.NPast;Verify(id+"/verify8",pass,slot,seed.Window);e.Commit(new[]{slot},8,new[]{keep});commits++;
                        Need(slot.NPast==pos+keep,"Commit position differs.");Checkpoint(id,pass,"committed",slot);
                        Decode(id+"/decode",pass,slot,seed.NextToken);Checkpoint(id,pass,"decoded",slot);
                        Proposal(id+"/proposal",pass,slot,seed.NextAnchor);Checkpoint(id,pass,"proposed",slot);
                        if(!pending && seed.Prompt.Length==255 && keep==8)
                        {
                            var copy=Clone(slot);try
                            {
                                var sourceHead=Decode(id+"/clone-source",pass,slot,seed.CloneToken);var copyHead=Decode(id+"/clone-copy",pass,copy,seed.CloneToken);
                                Need(MemoryMarshal.AsBytes(sourceHead.AsSpan(0,Vocab)).SequenceEqual(MemoryMarshal.AsBytes(copyHead.AsSpan(0,Vocab))),"Clone next-token full head differs.");
                                var a=State.Capture(e,slot,slot.NPast);var b=State.Capture(e,copy,copy.NPast);
                                Need(Encoded(State.Numeric(a))==Encoded(State.Numeric(b)),"Clone next-token effective state differs.");
                                Checkpoint(id,pass,"clone-source",slot);Checkpoint(id,pass,"clone-copy",copy);
                            }finally{e.Active=slot;e.FreeSlot(copy);}
                        }
                    }finally{e.Active=seedSlot;e.FreeSlot(slot);}
                }
                long firstPassCandidateCaptures=0;
                for(int pass=0;pass<2;pass++)
                {
                    foreach(var seed in seeds)
                    {
                        e.ResetSlot(seedSlot);resets++;e.Active=seedSlot;e.PrefillBatch(new[]{seedSlot},new[]{seed.Prompt},new float[1][]);e.Synchronize();Need(seedSlot.NPast==seed.Prompt.Length,"Seed prefill position differs.");
                        var before=State.Capture(e,seedSlot,seedSlot.NPast);
                        foreach(int keep in Protocol.Keeps)Case(seed,keep,false,pass);
                        if(seed.Prompt.Length==255)Case(seed,1,true,pass);
                        Need(Encoded(State.Numeric(before))==Encoded(State.Numeric(State.Capture(e,seedSlot,seedSlot.NPast))),"Cloned campaign mutated seed checkpoint.");
                    }
                    if(pass==0)
                    {
                        firstPassCandidateCaptures=State.Counters(e).Capture??0;var large=e.CreateSlot(0);
                        try{e.Active=large;e.PrefillBatch(new[]{large},new[]{pool.Prompt.Take(2048).ToArray()},new float[1][]);e.Synchronize();Need(large.NPast==2048,"Large prefill incomplete.");largePrefills++;}
                        finally{e.Active=seedSlot;e.FreeSlot(large);}
                    }
                }
                var final=State.Counters(e);report["finalCounters"]=final;
                if(mode=="on")Need(firstPassCandidateCaptures==1&&final.Capture==1&&final.Replay==20&&final.Completed==20&&final.Gemm==125&&final.Plain==144&&final.Initial==1&&final.LegacyPost==3,"Full mandatory candidate capture/reuse/normalization counts differ.");
                Need(verify8==20&&verify1==2&&commits==22&&decodes==24&&proposals==20&&clones==22&&resets==6&&largePrefills==1,"Mandatory operation campaign incomplete.");
                report["operations"]=new{verify8,verify1,commits,decodes,proposals,clones,resets,largePrefills};
                report["limits"]=new[]{"Same physical Verify1/8 histories only; no shortened-ordinary numerical oracle.","Complete head/tap raw bytes; valid Conv/State/KV/ring/pending digests with auditable offsets; future KV/padding and never-used RingPos excluded.","No task quality, performance, sampler distribution or broad device qualification claimed."};
            }
            // Complete only after all device resources/context have been disposed.
            Need(store.HeadRows==Protocol.HeadRows&&store.TapCaptures==22&&store.Vectors.Count==348&&store.RepeatChecks==174&&store.Bytes<=Protocol.RawBound,"Complete raw output campaign differs.");
            CheckPins(pins);store.VerifyFiles();artifactInfo.Refresh();Need(artifactInfo.Length==artifactStat.Length&&artifactInfo.LastWriteTimeUtc==artifactStat.LastWriteTimeUtc,"Artifact changed during producer.");
            using(var reader=new NinferArtifactReader(artifact))Need(reader.DirectorySha256==directorySha,"Artifact directory changed.");
            report["headRows"]=store.HeadRows;report["tapCaptures"]=store.TapCaptures;report["rawFilesBytes"]=store.Bytes;report["resetRawChecks"]=store.RepeatChecks;
            report["complete"]=true;report["passed"]=true;report["GPU"]="EXECUTED";Save(Path.Combine(output,"result.json"),report);Save(Path.Combine(output,"completed-result.json"),report,true);
            Console.WriteLine(Encoded(new{complete=true,mode,store.HeadRows,store.TapCaptures,store.Bytes,performance="NOT_MEASURED",quality="NOT_RUN"}));return 0;
        }
        catch(Exception error)
        {
            report["error"]=error.ToString();if(owned&&output!=null)Save(Path.Combine(output,"result.json"),report);Console.Error.WriteLine(error);return 1;
        }
    }
}
