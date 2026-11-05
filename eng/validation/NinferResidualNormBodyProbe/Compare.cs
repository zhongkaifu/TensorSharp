// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using static Support;

internal static class Compare
{
    private sealed record Evidence(string Folder,JsonElement Root,Dictionary<string,JsonElement> Vectors,Dictionary<string,string> States);
    private static string Id(JsonElement x) => x.GetProperty("id").GetString()+"/"+x.GetProperty("pass").GetInt32()+"/"+x.GetProperty("row").GetInt32()+"/"+x.GetProperty("format").GetString();
    private static Evidence Read(string directory,string mode)
    {
        Need(Path.IsPathFullyQualified(directory)&&directory==Path.GetFullPath(directory),"Canonical producer directory required.");
        string path=Path.Combine(directory,"completed-result.json");byte[] bytes=File.ReadAllBytes(path);
        string[] sidecar=File.ReadAllText(path+".sha256").Trim().Split(' ',StringSplitOptions.RemoveEmptyEntries);
        Need(sidecar.Length==2&&sidecar[0]==Hash(bytes)&&sidecar[1]=="completed-result.json","Completed certificate SHA sidecar differs.");
        using var doc=JsonDocument.Parse(bytes);Protocol.Unique(doc.RootElement);var r=doc.RootElement.Clone();
        Need(r.GetProperty("schema").GetString()=="ninfer-residual-norm-body/1"&&r.GetProperty("complete").GetBoolean()&&r.GetProperty("passed").GetBoolean()
            &&r.GetProperty("GPU").GetString()=="EXECUTED"&&r.GetProperty("mode").GetString()==mode&&r.GetProperty("headRows").GetInt32()==326
            &&r.GetProperty("tapCaptures").GetInt32()==22&&r.GetProperty("resetRawChecks").GetInt32()==174,"Mandatory completed producer scope differs.");
        Need(r.GetProperty("loadedImages").GetProperty("main").GetString()==(mode=="old"?Protocol.OldMain:Protocol.NewMain),"Actual loaded main image differs.");
        Need(FilePin(Path.Combine(directory,"source-inputs.json")).Sha256==r.GetProperty("sourceInputSha256").GetString()
            &&FilePin(Path.Combine(directory,"derived-inputs.json")).Sha256==r.GetProperty("derivedInputSha256").GetString(),"Retained original/derived inputs changed.");
        var operations=r.GetProperty("operations");
        foreach(var (key,value) in new Dictionary<string,int>{{"verify8",20},{"verify1",2},{"commits",22},{"decodes",24},{"proposals",20},{"clones",22},{"resets",6},{"largePrefills",1}})
            Need(operations.GetProperty(key).GetInt32()==value,"Incomplete operation campaign: "+key);
        Need(r.GetProperty("events").GetArrayLength()==22&&r.GetProperty("arguments").GetArrayLength()==66,"Mandatory call evidence differs.");
        var vectors=new Dictionary<string,JsonElement>(StringComparer.Ordinal);var files=new HashSet<string>();long total=0;int rows=0,taps=0;
        foreach(var v in r.GetProperty("vectors").EnumerateArray())
        {
            string format=v.GetProperty("format").GetString()!,sha=v.GetProperty("sha256").GetString()!,file=v.GetProperty("file").GetString()!;int n=v.GetProperty("bytes").GetInt32();
            Need((format=="f32"&&n==Vocab*4)||(format=="bf16"&&n is 51200 or 409600),"Raw output type/length differs.");
            Need(file=="vectors/"+sha+"."+format&&sha.Length==64&&vectors.TryAdd(Id(v),v.Clone()),"CAS path or unique vector identity differs.");
            var pin=FilePin(Path.Combine(directory,file));Need(pin.Bytes==n&&pin.Sha256==sha,"Retained complete raw bytes changed.");
            if(files.Add(file))total+=n;if(format=="f32")rows++;else taps++;
        }
        Need(vectors.Keys.ToHashSet().SetEquals(Protocol.VectorKeys())&&vectors.Count==348&&rows==326&&taps==22&&total==r.GetProperty("rawFilesBytes").GetInt64()&&total<=Protocol.RawBound,"Complete raw inventory differs.");
        var states=new Dictionary<string,string>(StringComparer.Ordinal);
        foreach(var x in r.GetProperty("checkpoints").EnumerateArray())
        {
            var s=JsonSerializer.Deserialize<State.Snapshot>(x.GetProperty("snapshot").GetRawText(),Json)!;
            Need(s.Parts.Length>=96&&s.WrittenKeys==s.Position&&s.Parts.All(p=>p.ByteOffset>=0&&p.Bytes>0&&p.Sha256.Length==64)
                &&s.Parts.Select(p=>p.Role).Distinct().Count()==s.Parts.Length,"Written-state slice inventory differs.");
            string key=x.GetProperty("id").GetString()+"/"+x.GetProperty("pass").GetInt32()+"/"+x.GetProperty("phase").GetString();
            Need(states.TryAdd(key,Encoded(State.Numeric(s))),"Duplicate effective-state checkpoint.");
        }
        Need(states.Keys.ToHashSet().SetEquals(Protocol.StateKeys())&&states.Count==86,"All mandatory state checkpoints required.");return new(directory,r,vectors,states);
    }
    private static object Pair(Evidence a,Evidence b)
    {
        foreach(string key in new[]{"artifactDirectorySha256","sourceInputSha256","derivedInputSha256","context","ubatch","numericalEnvironment","operations","arguments"})
            Need(Encoded(a.Root.GetProperty(key))==Encoded(b.Root.GetProperty(key)),"Same-history contract differs: "+key);
        foreach(string role in new[]{"helper","attention","a16","draft","repack","w8"})
            Need(a.Root.GetProperty("loadedImages").GetProperty(role).GetString()==b.Root.GetProperty("loadedImages").GetProperty(role).GetString(),"Shared arithmetic image differs: "+role);
        Need(a.Root.GetProperty("draftPayloadSha256").GetString()==b.Root.GetProperty("draftPayloadSha256").GetString(),"Original draft payload differs.");
        Need(a.Vectors.Keys.ToHashSet().SetEquals(b.Vectors.Keys)&&a.States.Keys.ToHashSet().SetEquals(b.States.Keys),"Mandatory output/state keys differ.");
        long compared=0;
        foreach(var (key,x) in a.Vectors)
        {
            var y=b.Vectors[key];byte[] av=File.ReadAllBytes(Path.Combine(a.Folder,x.GetProperty("file").GetString()!)),bv=File.ReadAllBytes(Path.Combine(b.Folder,y.GetProperty("file").GetString()!));
            Need(av.AsSpan().SequenceEqual(bv),"Full raw bytes differ: "+key);compared+=av.Length;
        }
        foreach(var (key,value) in a.States)Need(value==b.States[key],"Valid written-state bytes/metadata differ: "+key);
        return new{passed=true,from=a.Root.GetProperty("mode").GetString(),to=b.Root.GetProperty("mode").GetString(),fullRawRecordsCompared=a.Vectors.Count,fullHeadRows=326,tapCaptures=22,rawBytesCompared=compared,effectiveStateCheckpoints=a.States.Count};
    }
    internal static void Run(Dictionary<string,string> options)
    {
        string output=Required(options,"--out");Fresh(output);
        var old=Read(Required(options,"--old"),"old");var off=Read(Required(options,"--off"),"off");var on=Read(Required(options,"--on"),"on");
        // Baseline/new-OFF must agree before the candidate is credited.
        object oldOff=Pair(old,off),offOn=Pair(off,on);Directory.CreateDirectory(output);
        Save(Path.Combine(output,"completed-comparison.json"),new{schema="ninfer-residual-norm-body-comparison/1",complete=true,passed=true,
            producers=new[]{old,off,on}.Select(x=>FilePin(Path.Combine(x.Folder,"completed-result.json"))).ToArray(),oldOff,offOn,performance="NOT_MEASURED",quality="NOT_RUN"},true);
        Console.WriteLine(Encoded(new{complete=true,passed=true,oldOff,offOn}));
    }
}
