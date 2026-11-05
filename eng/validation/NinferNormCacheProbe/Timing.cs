// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    private static double Median(List<double> values){var a=values.Order().ToArray();return(a[a.Length/2-1]+a[a.Length/2])/2;}
    private static void RunTiming(Fixture fixture,Norm old,Norm candidate,Stream stream,string output,Dictionary<string,string> files,Dictionary<string,object?> state,Action checkpoint)
    {
        const string mode="disjoint";const int replays=64;using var memory=new Memory();var delta=memory.Put("timing-delta",Bytes(fixture.Delta));var weight=memory.Put("timing-weight",Bytes(fixture.Weight));
        var seedX=memory.Put("immutable-reset-X",Bytes(fixture.X));var initialY=new byte[Count*4];Array.Fill(initialY,(byte)0xd2);var seedY=memory.Put("immutable-reset-output",initialY);
        var left=new Arm(memory,"timing-legacy",mode,delta);var right=new Arm(memory,"timing-candidate",mode,delta);var oracle=Reference(fixture,mode);byte[] residual=Bytes(oracle.Residual);
        // Establish actual same-input direct outputs before capture. Each graph
        // then resets BOTH mutable buffers from immutable device seeds every
        // replay, so repeated residual mutations cannot change the workload.
        left.Reset(fixture,mode,0xc1);right.Reset(fixture,mode,0xc1);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
        old.Run(left.X.Pointer,left.Delta,weight.Pointer,left.Y.Pointer,stream.Pointer);candidate.Run(right.X.Pointer,right.Delta,weight.Pointer,right.Y.Pointer,stream.Pointer);stream.Sync();
        byte[] qualified=Memory.Body(left.Y);Need(qualified.AsSpan().SequenceEqual(Memory.Body(right.Y))&&Memory.Body(left.X).AsSpan().SequenceEqual(residual)&&Memory.Body(right.X).AsSpan().SequenceEqual(residual)&&Check(qualified,oracle).Passed,"Direct timing inputs not qualified.");
        using var control=new Captured(old,left,weight,seedX,seedY,stream.Pointer);using var test=new Captured(candidate,right,weight,seedX,seedY,stream.Pointer);
        var intervals=new List<object>();var oldSamples=new List<double>();var candidateSamples=new List<double>();var wins=new List<bool>();
        var record=new Dictionary<string,object?>{["passed"]=false,["scope"]="Captured twoD2Dresets+norm, both resets included symmetrically in every timed replay. This is neither isolated norm latency nor E2E throughput. Host graph submissions may contribute gaps; no cold-cache claim.",
            ["fixture"]=fixture.Name,["aliasMode"]=mode,["legacyNodes"]=control.Nodes,["candidateNodes"]=test.Nodes,["intervals"]=intervals,["timedReplaysPerInterval"]=replays,["minimumMedianSpeedupExclusive"]=1.05,["minimumPairedRoundWins"]=4};
        state["timing"]=record;checkpoint();
        for(int round=0;round<5;round++)
        {
            var sample=new double[4];for(int position=0;position<4;position++)
            {
                bool selected=position is 1 or 2;var graph=selected?test:control;var arm=selected?right:left;
                Memory.Poison(arm.X,Poisons[(round+position)%3]);Memory.Poison(arm.Y,Poisons[(round+position)%3]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                graph.Run(stream.Pointer);stream.Sync(); // Symmetric full-chain warmup outside events.
                CudaDriverApi.cuEventCreate(out var begin,0).ThrowOnError();IntPtr end=IntPtr.Zero;float elapsed;
                try
                {
                    CudaDriverApi.cuEventCreate(out end,0).ThrowOnError();CudaDriverApi.cuEventRecord(begin,stream.Pointer).ThrowOnError();for(int i=0;i<replays;i++)graph.Run(stream.Pointer);
                    CudaDriverApi.cuEventRecord(end,stream.Pointer).ThrowOnError();CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out elapsed,begin,end).ThrowOnError();
                }finally{if(end!=IntPtr.Zero)CudaDriverApi.cuEventDestroy(end).ThrowOnError();CudaDriverApi.cuEventDestroy(begin).ThrowOnError();}
                Need(float.IsFinite(elapsed)&&elapsed>0,"Invalid captured full-chain event interval.");byte[] x=Memory.Body(arm.X),y=Memory.Body(arm.Y);bool bits=x.AsSpan().SequenceEqual(residual)&&y.AsSpan().SequenceEqual(qualified),guards=memory.Guards(),inputs=memory.Inputs();var numeric=Check(y,oracle);
                var captured=new{residual=Cas(x,output,files),norm=Cas(y,output,files)};bool ok=bits&&guards&&inputs&&numeric.Passed;double perReplay=(double)elapsed/replays;sample[position]=perReplay;(selected?candidateSamples:oldSamples).Add(perReplay);
                intervals.Add(new{round,position,arm=selected?"candidate":"legacy",timedGraphReplays=replays,outsideWarmupReplays=1,intervalMilliseconds=elapsed,millisecondsPerResetAndNorm=perReplay,fullOutputBitsQualified=bits,allocationGuards=guards,immutableSeedsAndInputs=inputs,fp64=numeric,captured,passed=ok});checkpoint();Need(ok,"Timed reset+norm replay differs from qualified direct output.");
            }
            wins.Add((sample[1]+sample[2])/2<(sample[0]+sample[3])/2);
        }
        Need(intervals.Count==20&&oldSamples.Count==10&&candidateSamples.Count==10,"Five complete pairedABBA rounds mandatory.");double oldMedian=Median(oldSamples),candidateMedian=Median(candidateSamples),speedup=oldMedian/candidateMedian;
        record["passed"]=true;record["legacyMedianMilliseconds"]=oldMedian;record["candidateMedianMilliseconds"]=candidateMedian;record["medianSpeedup"]=speedup;record["pairedRoundWins"]=wins.Count(x=>x);record["roundWins"]=wins;
        record["predeclaredWin"]=speedup>1.05&&wins.Count(x=>x)>=4;record["actualTimedGraphReplays"]=1280;record["actualOutsideWarmupGraphReplays"]=20;record["actualNormExecutions"]=1300;record["actualD2DCopyExecutions"]=2600;
        record["directAdmissionNormExecutions"]=2;record["modelQuality"]="NOT_RUN";record["endToEndPerformance"]="NOT_RUN";checkpoint();
    }
}
