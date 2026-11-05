// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    private sealed record Qualified(byte[] Block,byte[] Partial,byte[] X,byte[] Y,NormOracle Oracle);
    private static double Median(List<double> values){var a=values.Order().ToArray();return(a[4]+a[5])/2;}
    private static void RunTiming(Modules m,Stream stream,string output,Dictionary<string,string> files,Dictionary<string,object?> state,Action checkpoint)
    {
        var results=new List<Dictionary<string,object?>>();state["timing"]=results;
        foreach(int addressCount in new[]{1,6})
        {
            var f=DenseTiming();using var memory=new Memory();var input=memory.Put("timing-input",Bytes(f.Input));float[] weights=NormWeights();var weight=memory.Put("norm-weights",Bytes(weights));var addresses=new Address[addressCount];var seeds=new float[addressCount][];var qualified=new Qualified[addressCount];
            var admission=new List<object>();var intervals=new List<object>();var wins=new List<bool>();var left=new List<double>();var right=new List<double>();
            var record=new Dictionary<string,object?>{["passed"]=false,["addressCount"]=addressCount,["scope"]="Complete reset_x/quant/GEMM/norm graph. One or six independent dense-weight allocations, repeated addresses; no physical-cold/DRAM/E2E claim. Same-source8 is excluded from timing.",["weightBytesEach"]=f.WeightCodes.Length+f.WeightScales.Length,["weightWorkingSetBytes"]=(long)addressCount*(f.WeightCodes.Length+f.WeightScales.Length),["projectionFp64WitnessValuesPerAddress"]=Rows*Active.Length,["normFp64ValuesPerAddress"]=OutputCount,["weightCodesSha256"]=Sha(f.WeightCodes),["weightScalesSha256"]=Sha(f.WeightScales),["minimumMedianSpeedupExclusive"]=1.05,["minimumPairedRoundWins"]=4,["timedGraphReplaysPerInterval"]=64,["admission"]=admission,["intervals"]=intervals};results.Add(record);checkpoint();
            try
            {
                for(int i=0;i<addressCount;i++)
                {
                    seeds[i]=Seed(f,i*13);addresses[i]=new(memory.Put("weights"+i+"-codes",f.WeightCodes),memory.Put("weights"+i+"-scales",f.WeightScales),memory.Put("seed"+i,Bytes(seeds[i])),new(memory,"old"+i),new(memory,"candidate"+i));
                    var a=addresses[i];a.Old.Reset(seeds[i],0xc1);a.New.Reset(seeds[i],0xd2);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                    m.Quantize(f,input,a.Old,stream.Pointer);m.Project(f,a.Wc,a.Ws,a.Old,0,stream.Pointer);m.Quantize(f,input,a.New,stream.Pointer);m.Project(f,a.Wc,a.Ws,a.New,2,stream.Pointer);stream.Sync();
                    byte[] block=Memory.Body(a.Old.Block),partial=Memory.Body(a.Old.Partial),candidateBefore=Memory.Body(a.New.X);var oracle=NormReference(seeds[i],block,weights);
                    bool before=block.AsSpan().SequenceEqual(Memory.Body(a.New.Block))&&partial.AsSpan().SequenceEqual(Memory.Body(a.New.Partial))&&candidateBefore.AsSpan().SequenceEqual(Bytes(oracle.Residual))&&Check(block,f,false).Passed;
                    m.Norm(a.Old,weight,0,stream.Pointer);m.Norm(a.New,weight,2,stream.Pointer);stream.Sync();byte[] x=Memory.Body(a.Old.X),y=Memory.Body(a.Old.Y);
                    bool after=x.AsSpan().SequenceEqual(Bytes(oracle.Residual))&&x.AsSpan().SequenceEqual(Memory.Body(a.New.X))&&candidateBefore.AsSpan().SequenceEqual(Memory.Body(a.New.X))&&y.AsSpan().SequenceEqual(Memory.Body(a.New.Y))&&CheckNorm(y,oracle).Passed&&a.Old.QuantBits(f)&&a.New.QuantBits(f)&&a.Old.CountersZero()&&a.New.CountersZero()&&memory.Guards()&&memory.Inputs();
                    qualified[i]=new(block,partial,x,y,oracle);admission.Add(new{address=i,fullBeforeBits=before,fullAfterBits=after,projectionFp64=Check(block,f,false),normFp64=CheckNorm(y,oracle),block=Cas(block,output,files),partial=Cas(partial,output,files),residual=Cas(x,output,files),norm=Cas(y,output,files),candidateBeforeNormResidual=Cas(candidateBefore,output,files),candidateBlock=Cas(Memory.Body(a.New.Block),output,files),candidatePartial=Cas(Memory.Body(a.New.Partial),output,files),candidateResidual=Cas(Memory.Body(a.New.X),output,files),candidateNorm=Cas(Memory.Body(a.New.Y),output,files),seedSha256=Sha(Bytes(seeds[i])),passed=before&&after});checkpoint();Need(before&&after,"Dense full-chain numerical admission failed; do not time.");
                }
                Need(addresses.Select(a=>a.Wc.Pointer).Distinct().Count()==addressCount&&addresses.Select(a=>a.Ws.Pointer).Distinct().Count()==addressCount,"Weight addresses are not independent allocations.");
                using var old=new ChainGraph(m,f,input,weight,addresses,false,stream.Pointer);using var candidate=new ChainGraph(m,f,input,weight,addresses,true,stream.Pointer);record["legacyGraphNodes"]=old.Nodes;record["candidateGraphNodes"]=candidate.Nodes;checkpoint();
                for(int round=0;round<5;round++)
                {
                    var sample=new double[4];for(int position=0;position<4;position++)
                    {
                        bool selected=position is 1 or 2;var graph=selected?candidate:old;
                        for(int i=0;i<addressCount;i++)(selected?addresses[i].New:addresses[i].Old).Reset(seeds[i],Poisons[(round+position)%3]);CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();graph.Run(stream.Pointer);stream.Sync();
                        CudaDriverApi.cuEventCreate(out var begin,0).ThrowOnError();IntPtr end=IntPtr.Zero;float elapsed;
                        try{CudaDriverApi.cuEventCreate(out end,0).ThrowOnError();CudaDriverApi.cuEventRecord(begin,stream.Pointer).ThrowOnError();for(int replay=0;replay<64;replay++)graph.Run(stream.Pointer);CudaDriverApi.cuEventRecord(end,stream.Pointer).ThrowOnError();CudaDriverApi.cuEventSynchronize(end).ThrowOnError();CudaDriverApi.cuEventElapsedTime(out elapsed,begin,end).ThrowOnError();}
                        finally{if(end!=IntPtr.Zero)CudaDriverApi.cuEventDestroy(end).ThrowOnError();CudaDriverApi.cuEventDestroy(begin).ThrowOnError();}
                        Need(float.IsFinite(elapsed)&&elapsed>0,"Invalid paired complete-chain CUDA event interval.");var checks=new List<object>();bool ok=true;
                        for(int i=0;i<addressCount;i++)
                        {
                            var arm=selected?addresses[i].New:addresses[i].Old;var q=qualified[i];byte[] block=Memory.Body(arm.Block),partial=Memory.Body(arm.Partial),x=Memory.Body(arm.X),y=Memory.Body(arm.Y);
                            bool bits=block.AsSpan().SequenceEqual(q.Block)&&partial.AsSpan().SequenceEqual(q.Partial)&&x.AsSpan().SequenceEqual(q.X)&&y.AsSpan().SequenceEqual(q.Y),quant=arm.QuantBits(f),counter=arm.CountersZero();var pn=Check(block,f,false);var nn=CheckNorm(y,q.Oracle);bool good=bits&&quant&&counter&&pn.Passed&&nn.Passed;ok&=good;
                            checks.Add(new{address=i,fullBlockPartialResidualNormBits=bits,cpuQuantBits=quant,countersReturnedZero=counter,projectionFp64Witness=pn,normFp64=nn,block=Cas(block,output,files),partial=Cas(partial,output,files),residual=Cas(x,output,files),norm=Cas(y,output,files),passed=good});
                        }
                        bool guards=memory.Guards(),inputs=memory.Inputs();ok&=guards&&inputs;double ms=(double)elapsed/64;sample[position]=ms;(selected?right:left).Add(ms);
                        intervals.Add(new{round,position,arm=selected?"fused8-plain-norm":"actual-old8-old-addnorm",intervalMilliseconds=elapsed,timedGraphReplays=64,outsideWarmupGraphReplays=1,millisecondsPerMatrixPass=ms,millisecondsPerProjectionAndNorm=ms/addressCount,addressChecks=checks,allocationGuards=guards,immutableInputs=inputs,passed=ok});checkpoint();Need(ok,"Timed chain changed full qualified outputs/inputs/guards.");
                    }
                    wins.Add((sample[1]+sample[2])/2<(sample[0]+sample[3])/2);
                }
                Need(intervals.Count==20&&left.Count==10&&right.Count==10,"All five ABBA blocks mandatory.");double aMs=Median(left),bMs=Median(right);record["passed"]=true;record["legacyMedianMillisecondsPerMatrixPass"]=aMs;record["candidateMedianMillisecondsPerMatrixPass"]=bMs;record["medianSpeedup"]=aMs/bMs;record["roundWins"]=wins;record["pairedRoundWins"]=wins.Count(v=>v);record["predeclaredWin"]=aMs/bMs>1.05&&wins.Count(v=>v)>=4;record["actualTimedGraphReplays"]=1280;record["actualOutsideWarmupGraphReplays"]=20;record["actualTimingKernelExecutionsSubmitted"]=1300*addressCount*3;record["actualTimingResetCopiesSubmitted"]=1300*addressCount;record["directAdmissionKernelExecutionsSubmitted"]=6*addressCount;record["captureHostKernelSubmissionsNotExecution"]=6*addressCount;checkpoint();
            }
            finally{stream.Sync();}
        }
        state["timingAllAddressModesPredeclaredWin"]=results.Count==2&&results.All(r=>r["predeclaredWin"] is true);checkpoint();
    }
}
