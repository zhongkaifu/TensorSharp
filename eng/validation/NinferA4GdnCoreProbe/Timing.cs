// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    private static double Median(List<double> x){var a=x.Order().ToArray();return(a[a.Length/2-1]+a[a.Length/2])/2;}
    private static void RunTiming(Modules modules,Stream stream,string output,Dictionary<string,string> files,Dictionary<string,object?> state,Action checkpoint)
    {
        const int replays=64;var f=DenseTiming();using var memory=new Memory();var x=memory.Put("timing-input",Bytes(f.Input));var wc=memory.Put("timing-weight-codes",f.WeightCodes);var ws=memory.Put("timing-weight-scales",f.WeightScales);GemmArm old=new(memory,"legacy"),candidate=new(memory,"candidate");
        old.Reset(0xc1);candidate.Reset(0xc1);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();modules.Quantize(f,x,old,stream.Pointer);modules.Project(f,wc,ws,old,0,stream.Pointer);modules.Quantize(f,x,candidate,stream.Pointer);modules.Project(f,wc,ws,candidate,2,stream.Pointer);stream.Sync();
        byte[] directF32=Memory.Body(old.Output);Need(directF32.AsSpan().SequenceEqual(Memory.Body(candidate.Output))&&Check(directF32,f,false).Passed,"Dense direct fullF32 bits/FP64 witness differ.");
        modules.Consume(old,stream.Pointer);modules.Consume(candidate,stream.Pointer);stream.Sync();byte[] qualified=Memory.Body(old.Final),qualifiedPartial=Geometry.Splits==1?Array.Empty<byte>():Memory.Body(old.Partial);
        Need(qualified.AsSpan().SequenceEqual(Memory.Body(candidate.Final))&&(Geometry.Splits==1||qualifiedPartial.AsSpan().SequenceEqual(Memory.Body(candidate.Partial)))&&old.QuantBits(f)&&candidate.QuantBits(f)&&ConsumerFormat(qualified).Passed,"Timing direct full chain not qualified.");
        using var a=new ChainGraph(modules,f,x,wc,ws,old,0,stream.Pointer);using var b=new ChainGraph(modules,f,x,wc,ws,candidate,2,stream.Pointer);var intervals=new List<object>();var wins=new List<bool>();var left=new List<double>();var right=new List<double>();
        var record=new Dictionary<string,object?>{["passed"]=false,["scope"]=$"Warm repeated dense pseudorandom weight addresses; not physicalcold/DRAM/E2E. {LegacyPlan.Arm} versus candidate8 only; same-source32 is excluded from timing. Complete real role's quant/GEMM/consumer chain. Alloutputs require exactGPUbits+finite; BF16 onlyforround/SwiGLU. FP64witness is beforeconsumer only, never upgraded to fullFP64coverage.",["legacySchedule"]=LegacyScheduleName,["legacyArm"]=LegacyPlan.Arm,["legacySymbol"]=LegacyPlan.Symbol,["legacyDynamicSharedBytes"]=LegacyPlan.Shared,["candidateSymbol"]=ProjectionPlan(2,LegacySchedule).Symbol,["geometry"]=Geometry,["weightBytes"]=f.WeightCodes.Length+f.WeightScales.Length,["fp64WitnessRows"]=Active.Length,["fp64WitnessValues"]=Rows*Active.Length,["fixture"]=f.Name,["weightCodesSha256"]=Sha(f.WeightCodes),["weightScalesSha256"]=Sha(f.WeightScales),["directAdmissionFullF32BitsExact"]=true,["directF32File"]=Cas(directF32,output,files),["directF32Fp64Witness"]=Check(directF32,f,false),["legacyNodes"]=a.Nodes,["candidateNodes"]=b.Nodes,["minimumMedianSpeedupExclusive"]=1.05,["minimumPairedRoundWins"]=4,["timedReplaysPerInterval"]=replays,["intervals"]=intervals};state["timing"]=record;checkpoint();
        for(int round=0;round<5;round++)
        {
            var sample=new double[4];for(int position=0;position<4;position++)
            {
                bool selected=position is 1 or 2;var graph=selected?b:a;var arm=selected?candidate:old;arm.Reset(Poisons[(round+position)%3]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();graph.Run(stream.Pointer);stream.Sync();
                CudaDriverApi.cuEventCreate(out var begin,0).ThrowOnError();IntPtr end=IntPtr.Zero;float elapsed;
                try{CudaDriverApi.cuEventCreate(out end,0).ThrowOnError();CudaDriverApi.cuEventRecord(begin,stream.Pointer).ThrowOnError();for(int i=0;i<replays;i++)graph.Run(stream.Pointer);CudaDriverApi.cuEventRecord(end,stream.Pointer).ThrowOnError();CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out elapsed,begin,end).ThrowOnError();}
                finally{if(end!=IntPtr.Zero)CudaDriverApi.cuEventDestroy(end).ThrowOnError();CudaDriverApi.cuEventDestroy(begin).ThrowOnError();}
                Need(float.IsFinite(elapsed)&&elapsed>0,"Invalid complete-chain event interval.");byte[] y=Memory.Body(arm.Final),p=Geometry.Splits==1?Array.Empty<byte>():Memory.Body(arm.Partial);byte[]? projection=Geometry.Consumer=="round"?null:Geometry.Consumer=="none"?y:Memory.Body(arm.Output);
                bool bits=y.AsSpan().SequenceEqual(qualified)&&(Geometry.Splits==1||p.AsSpan().SequenceEqual(qualifiedPartial))&&(projection==null||projection.AsSpan().SequenceEqual(directF32)),quant=arm.QuantBits(f),counter=Geometry.Splits==1||arm.CountersZero(),guards=memory.Guards(),inputs=memory.Inputs();var format=ConsumerFormat(y);var numeric=projection==null?null:Check(projection,f,false);bool ok=bits&&quant&&counter&&guards&&inputs&&format.Passed&&(numeric?.Passed??true);double ms=(double)elapsed/replays;sample[position]=ms;(selected?right:left).Add(ms);
                intervals.Add(new{round,position,arm=selected?"candidate8":LegacyPlan.Arm,timedGraphReplays=replays,outsideWarmupReplays=1,intervalMilliseconds=elapsed,millisecondsPerFullChain=ms,fullQualifiedBits=bits,partialBitsChecked=Geometry.Splits>1,unusedScratchGuardsOnly=Geometry.Splits==1,projectionF32BitsChecked=projection!=null,quantizedPayloadBits=quant,countersReturnedZero=Geometry.Splits>1?(bool?)counter:null,allocationGuards=guards,immutableInputs=inputs,consumerFormat=format,fp64BeforeConsumer=numeric,outputFile=Cas(y,output,files),projectionFile=projection!=null?Cas(projection,output,files):null,partialFile=Geometry.Splits>1?Cas(p,output,files):null,passed=ok});checkpoint();Need(ok,"Timed full chain changed qualified payload/output/guards.");
            }
            wins.Add((sample[1]+sample[2])/2<(sample[0]+sample[3])/2);
        }
        Need(intervals.Count==20&&left.Count==10&&right.Count==10,"All fiveABBA blocks required.");double lm=Median(left),rm=Median(right);record["passed"]=true;record["legacyMedianMilliseconds"]=lm;record["candidateMedianMilliseconds"]=rm;record["medianSpeedup"]=lm/rm;record["pairedRoundWins"]=wins.Count(v=>v);record["roundWins"]=wins;record["predeclaredWin"]=lm/rm>1.05&&wins.Count(v=>v)>=4;
        record["actualTimedGraphReplays"]=1280;record["actualOutsideWarmupGraphReplays"]=20;record["actualTimingKernelExecutions"]=1300*NodesPerChain;record["directAdmissionKernelExecutions"]=2*NodesPerChain;checkpoint();
    }
}
