// Standalone current-layer exact F16 intermediate. Every candidate call times
// predecode + original128-query attention + combine. No production integration.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "checks.h"
#include "half_checks.h"

namespace fs = std::filesystem;
static void Cu(CUresult e) { if(e != CUDA_SUCCESS) { const char* s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"CUDA driver failure"); } }
static void Rt(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
constexpr int D=256, B=256, Q8=272, Guard=64;
constexpr uint32_t Canary=0x4f123456;
struct Shape { int prefix, rows, seq, g, kv, splits, pattern; };
static std::string Name(const Shape& s) { return "p"+std::to_string(s.prefix)+"-r"+std::to_string(s.rows)+"-s"+std::to_string(s.seq)+"-g"+std::to_string(s.g)+"-k"+std::to_string(s.kv)+"-x"+std::to_string(s.splits)+"-v"+std::to_string(s.pattern); }
struct Allocation {
    void* base=nullptr;size_t bytes=0;
    explicit Allocation(size_t n):bytes(n) { Rt(cudaMalloc(&base,n)); }
    Allocation(const Allocation&)=delete;
    ~Allocation() { if(base) cudaFree(base); }
};
struct Variant {
    int id,shared;bool permitted=true;CUmodule module=nullptr;
    CUfunction attention=nullptr,combine=nullptr,decode=nullptr,reference=nullptr;
    bool Inspect(CUfunction function,const char* name,int dynamic) {
        int regs=0,local=0,stat=0,active=0,maxThreads=0;
        Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
        Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
        Cu(cuFuncGetAttribute(&stat,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
        Cu(cuFuncGetAttribute(&maxThreads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function));
        if(dynamic)Cu(cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,dynamic));
        Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,256,dynamic));
        std::printf("{\"event\":\"device_resources\",\"variant\":%d,\"symbol\":\"%s\",\"regs\":%d,\"local_bytes\":%d,\"dynamic_shared\":%d,\"static_shared\":%d,\"potential_active_ctas\":%d,\"max_threads\":%d,\"two_cta_required\":false}\n",id,name,regs,local,dynamic,stat,active,maxThreads);
        // Spills/local memory are reported facts, as for the unmodified oracle.
        // This experiment requires launchability, not the separate64-query gate.
        return regs>0&&regs<=255&&maxThreads>=256&&active>=1;
    }
    Variant(int v,const fs::path& dir):id(v),shared(v==0?68608:67584) {
        Cu(cuModuleLoad(&module,(dir/"kernels.cubin").c_str()));
        const char* name=v==0?"ts_bench_attn_prefill":v==1?"ts_bench_attn_f16_prefill":"ts_bench_attn_f16_no_second_barrier";
        Cu(cuModuleGetFunction(&attention,module,name));
        Cu(cuModuleGetFunction(&combine,module,"ts_bench_attn_combine"));
        permitted=Inspect(attention,name,shared);
        if(v>0) {
            Cu(cuModuleGetFunction(&decode,module,"ts_bench_decode_plain"));
            Cu(cuModuleGetFunction(&reference,module,"ts_bench_decode_reference"));
            permitted=Inspect(decode,"ts_bench_decode_plain",0)&&permitted;
            permitted=Inspect(reference,"ts_bench_decode_reference",0)&&permitted;
            permitted=Inspect(combine,"ts_bench_attn_combine",0)&&permitted;
        }
        if(!permitted)std::printf("{\"event\":\"resource_rejected_no_gpu_run\",\"variant\":%d}\n",id);
    }
    ~Variant() { if(module) cuModuleUnload(module); }
};
struct Result { std::vector<uint32_t> output,partial; };
static void Save(const fs::path& p,const std::vector<uint32_t>& a);
struct Fixture {
    Shape s;int heads,keys,blocks;size_t rowHeads,qn,pn,blockBytes;long long ko=0,vo;
    Allocation cache,q,gate,ends,table,out,partial,decoded;
    std::vector<int> endHost;std::vector<void*> pageHost;unsigned resetCount=0;
    size_t HalfCount(int stride)const{return(size_t)s.seq*2*s.kv*keys*stride;}
    void* HalfData(Allocation& buffer)const{return(uint8_t*)buffer.base+Guard*4;}
    void Poison(Allocation& buffer,size_t count,unsigned char byte) {
        Rt(cudaMemset(buffer.base,0x56,buffer.bytes));
        Rt(cudaMemset(HalfData(buffer),byte,count*2));
    }
    void HalfGuards(Allocation& buffer) {
        std::array<uint32_t,Guard> a{},b{};Rt(cudaMemcpy(a.data(),buffer.base,Guard*4,cudaMemcpyDeviceToHost));
        Rt(cudaMemcpy(b.data(),(uint8_t*)buffer.base+buffer.bytes-Guard*4,Guard*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<Guard;i++)if(a[i]!=0x56565656u||b[i]!=0x56565656u)throw std::runtime_error("F16 workspace guard corruption");
    }
    std::vector<uint16_t> HalfRead(Allocation& buffer,size_t count) {
        Rt(cudaDeviceSynchronize());HalfGuards(buffer);std::vector<uint16_t> data(count);
        Rt(cudaMemcpy(data.data(),HalfData(buffer),count*2,cudaMemcpyDeviceToHost));return data;
    }
    void Decode(Variant& v,Allocation& destination,bool reference,void* alternateEnds=nullptr,void* alternateTable=nullptr) {
        void* tp=alternateTable?alternateTable:table.base;void* ep=alternateEnds?alternateEnds:ends.base;void* dst=HalfData(destination);
        int mb=blocks,rows=s.rows,nk=s.kv,capacity=keys;long long koff=ko,voff=vo;
        void* args[]={&tp,&mb,&koff,&voff,&ep,&rows,&capacity,&nk,&dst};
        Cu(cuLaunchKernel(reference?v.reference:v.decode,(keys+31)/32,s.kv,s.seq*2,256,1,1,0,nullptr,args,nullptr));
    }
    void CompareHalves(const std::vector<uint16_t>& expected,const std::vector<uint16_t>& actual,
                       int actualStride,uint16_t expectedPoison,uint16_t actualPoison,const std::vector<int>& validEnds,const fs::path& output) {
        if(auto fail=FindDecodedHalfMismatch(expected,actual,s.seq,s.kv,keys,validEnds,actualStride,expectedPoison,actualPoison)) {
            std::ofstream a(output/"failure-decode-reference.u16",std::ios::binary),b(output/"failure-decode-actual.u16",std::ios::binary);
            a.write((const char*)expected.data(),expected.size()*2);b.write((const char*)actual.data(),actual.size()*2);
            std::ofstream info(output/"failure-decode.json");info<<"{\"shape\":\""<<Name(s)<<"\",\"row\":"<<fail->row<<",\"column\":"<<fail->column<<",\"actual_stride\":"<<actualStride<<"}\n";
            throw std::runtime_error("complete decoded half-bit/write-coverage mismatch");
        }
    }
    void VerifyDecodePass(Variant& v,const fs::path& output,const std::vector<int>& validEnds,void* alternateEnds=nullptr,void* alternateTable=nullptr) {
        // A legitimate half may equal ANY sentinel, including a NaN. Compare
        // all valid bits and use distinct poison passes, never value exclusions.
        Allocation reference(HalfCount(264)*2+2*Guard*4);
        Poison(reference,HalfCount(264),0x3c);Decode(v,reference,true,alternateEnds,alternateTable);auto first=HalfRead(reference,HalfCount(264));
        Poison(reference,HalfCount(264),0xc3);Decode(v,reference,true,alternateEnds,alternateTable);auto second=HalfRead(reference,HalfCount(264));
        CompareHalves(first,second,264,0x3c3c,0xc3c3,validEnds,output);std::vector<uint16_t>().swap(second);
        for(unsigned char color:{(unsigned char)0xa5,(unsigned char)0x7e}) {
            Poison(decoded,HalfCount(256),color);Decode(v,decoded,false,alternateEnds,alternateTable);auto actual=HalfRead(decoded,HalfCount(256));
            CompareHalves(first,actual,256,0x3c3c,uint16_t(color)*0x101,validEnds,output);
        }
        std::printf("{\"event\":\"decoded_half_bits_pass\",\"shape\":\"%s\",\"workspace_bytes\":%zu,\"current_layer_only\":true,\"legacy_reference_AA\":true,\"candidate_two_poison\":true,\"unused_key_tail_probe\":%s}\n",Name(s).c_str(),HalfCount(256)*2,alternateEnds?"true":"false");
        // Strict-only padded reference storage is released before attention/timing.
    }
    bool VerifyDecode(Variant& v,const fs::path& output) {
        std::vector<int> validEnds(s.seq);for(int seq=0;seq<s.seq;seq++)validEnds[seq]=endHost[(size_t)seq*s.rows+s.rows-1];
        VerifyDecodePass(v,output,validEnds);
        if(keys<=512) {
            // Exercise allocated-but-unused keys and unequal per-lane bounds
            // without mutating the attention fixture's real causal ends.
            std::vector<int> limited=endHost;std::vector<void*> limitedPages=pageHost;
            for(int seq=0;seq<s.seq;seq++) {
                validEnds[seq]=seq==0?std::max(0,keys-7):keys;
                for(int t=0;t<s.rows;t++)limited[(size_t)seq*s.rows+t]=std::min(limited[(size_t)seq*s.rows+t],validEnds[seq]);
                for(int b=(validEnds[seq]+B-1)/B;b<blocks;b++)limitedPages[(size_t)seq*blocks+b]=nullptr;
            }
            Allocation alternate(limited.size()*sizeof(int));Rt(cudaMemcpy(alternate.base,limited.data(),alternate.bytes,cudaMemcpyHostToDevice));
            Allocation alternateTable(limitedPages.size()*sizeof(void*));Rt(cudaMemcpy(alternateTable.base,limitedPages.data(),alternateTable.bytes,cudaMemcpyHostToDevice));
            VerifyDecodePass(v,output,validEnds,alternate.base,alternateTable.base);
            return true;
        }
        return false;
    }
    Fixture(Shape shape):s(shape),heads(s.g*s.kv),keys(s.prefix+s.rows),blocks((keys+B-1)/B),rowHeads((size_t)s.seq*s.rows*heads),
        qn(rowHeads*D),pn(rowHeads*s.splits*(D+2)),blockBytes(2ull*s.kv*B*Q8),vo((long long)s.kv*B*Q8),
        cache((size_t)s.seq*blocks*blockBytes+2*Guard*4),q(qn*4),gate(qn*4),ends((size_t)s.seq*s.rows*4),
        table((size_t)s.seq*blocks*sizeof(void*)),out((qn+2*Guard)*4),partial((pn+2*Guard)*4),
        decoded((size_t)s.seq*2*s.kv*keys*D*2+2*Guard*4) {
        std::mt19937 rng(12347+s.prefix+s.rows*17+s.pattern*67);std::uniform_int_distribution<int> code(-127,127);
        std::uniform_real_distribution<float> f(-2,2);
        std::vector<uint8_t> bytes(cache.bytes,0x56);size_t body=(size_t)s.seq*blocks*blockBytes;
        uint8_t* raw=bytes.data()+Guard*4;
        for(size_t row=0;row<body/Q8;row++) {
            uint8_t* p=raw+row*Q8;
            for(int d=0;d<D;d++) p[d]=(uint8_t)(int8_t)(s.pattern==2?0:code(rng));
            for(int j=0;j<8;j++) {
                half h=__float2half(.03125f*(1+(row+j)%4));uint16_t bits;std::memcpy(&bits,&h,2);
                if(s.pattern==1) { const uint16_t x[]={0,0x8000,1,0x8001,0x1400,0x9400,0x2800,0xa800};bits=x[(row+j)%8]; }
                if(s.pattern==3) { bits=0x2800;for(int d=j*32;d<(j+1)*32;d++)p[d]=0;p[j*32]=(uint8_t)(int8_t)((int)(row%251)-125); }
                if(s.pattern==4 && (row+j)%11==0) bits=(row&1)?0x7c00:0x7e01;
                std::memcpy(p+256+2*j,&bits,2);
            }
        }
        std::vector<float> hostQ(qn),hostGate(qn);
        for(size_t i=0;i<qn;i++) { hostQ[i]=s.pattern==2?0:f(rng);hostGate[i]=f(rng); }
        if(s.pattern==3) for(int seq=0;seq<s.seq;seq++)for(int t=0;t<s.rows;t++)for(int h=0;h<heads;h++) {
            size_t at=(((size_t)seq*s.rows+t)*heads+h)*D;std::fill_n(&hostQ[at],D,0);
            int logical=t*s.g+h%s.g;hostQ[at]=(logical%16<8?1:-1)*100.0f;
        }
        std::vector<int> hostEnds((size_t)s.seq*s.rows);for(int seq=0;seq<s.seq;seq++)for(int t=0;t<s.rows;t++)hostEnds[(size_t)seq*s.rows+t]=s.prefix+t+1;
        std::vector<int> order((size_t)s.seq*blocks);for(size_t i=0;i<order.size();i++)order[i]=(int)i;
        std::shuffle(order.begin(),order.end(),rng);
        std::vector<void*> pointers(order.size());for(size_t i=0;i<order.size();i++) pointers[i]=(uint8_t*)cache.base+Guard*4+(size_t)order[i]*blockBytes;
        // Shared immutable prefix pages, distinct last page (COW-style) for other lanes.
        if(s.seq>1)for(int seq=1;seq<s.seq;seq++)for(int k=0;k<s.prefix/B;k++)pointers[(size_t)seq*blocks+k]=pointers[k];
        // Predecode's last-key bound is valid only for monotone causal ends and
        // resident pages through that end. State the same contract explicitly.
        if((long long)keys>(long long)blocks*B)throw std::runtime_error("key capacity exceeds page table");
        for(int seq=0;seq<s.seq;seq++) {
            int previous=0,limit=hostEnds[(size_t)seq*s.rows+s.rows-1];
            if(limit<0||limit>keys)throw std::runtime_error("invalid final sequence key end");
            for(int t=0;t<s.rows;t++){int end=hostEnds[(size_t)seq*s.rows+t];if(end<previous||end>limit)throw std::runtime_error("nonmonotone causal key ends");previous=end;}
            for(int b=0;b<(limit+B-1)/B;b++)if(!pointers[(size_t)seq*blocks+b])throw std::runtime_error("missing visible KV page");
        }
        Rt(cudaMemcpy(cache.base,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        Rt(cudaMemcpy(q.base,hostQ.data(),qn*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(gate.base,hostGate.data(),qn*4,cudaMemcpyHostToDevice));
        Rt(cudaMemcpy(ends.base,hostEnds.data(),hostEnds.size()*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(table.base,pointers.data(),pointers.size()*sizeof(void*),cudaMemcpyHostToDevice));
        endHost=hostEnds;pageHost=pointers;
    }
    void Reset() {
        std::vector<uint32_t> a(qn+2*Guard,Canary),b(pn+2*Guard,Canary);
        Rt(cudaMemcpy(out.base,a.data(),a.size()*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(partial.base,b.data(),b.size()*4,cudaMemcpyHostToDevice));
        Poison(decoded,HalfCount(256),++resetCount%2?0xa5:0x7e);
    }
    void Issue(Variant& v) {
        void *qp=q.base,*gp=gate.base,*ep=ends.base,*tp=table.base,*pp=(uint32_t*)partial.base+Guard,*op=(uint32_t*)out.base+Guard;
        int maxBlocks=blocks,rows=s.rows,nh=heads,nk=s.kv,sp=s.splits;float scale=1/std::sqrt((float)D);long long koff=ko,voff=vo;
        unsigned tiles=(rows*s.g+127)/128,physical=tiles*sp;
        if(v.id==0) {
            void* args[]={&qp,&gp,&ep,&tp,&maxBlocks,&koff,&voff,&rows,&nh,&nk,&scale,&sp,&pp,&op};
            Cu(cuLaunchKernel(v.attention,physical,nk,s.seq,256,1,1,v.shared,nullptr,args,nullptr));
        } else {
            Decode(v,decoded,false); // Included in EVERY candidate call and event interval.
            void* halfp=HalfData(decoded);int capacity=keys;
            void* args[]={&qp,&gp,&ep,&halfp,&capacity,&rows,&nh,&nk,&scale,&sp,&pp,&op};
            Cu(cuLaunchKernel(v.attention,physical,nk,s.seq,256,1,1,v.shared,nullptr,args,nullptr));
        }
        if(sp>1) { void* merge[]={&pp,&gp,&sp,&op};Cu(cuLaunchKernel(v.combine,(unsigned)rowHeads,1,1,256,1,1,0,nullptr,merge,nullptr)); }
    }
    Result Read(const fs::path& output) {
        Rt(cudaDeviceSynchronize());Result r;r.output.resize(qn+2*Guard);r.partial.resize(pn+2*Guard);
        Rt(cudaMemcpy(r.output.data(),out.base,r.output.size()*4,cudaMemcpyDeviceToHost));Rt(cudaMemcpy(r.partial.data(),partial.base,r.partial.size()*4,cudaMemcpyDeviceToHost));
        try {
            CheckGuardedWords(r.output,qn,Guard,Canary,true,"output");
            CheckGuardedWords(r.partial,pn,Guard,Canary,s.splits>1,"partial");
        } catch(const std::exception&) {
            Save(output/"failure-write-coverage-output.f32",r.output);Save(output/"failure-write-coverage-partial.f32",r.partial);
            std::ofstream info(output/"failure.json");info<<"{\"shape\":\""<<Name(s)<<"\",\"reason\":\"guard_or_write_coverage_failure\"}\n";
            throw;
        }
        std::array<uint32_t,Guard> first{},last{};Rt(cudaMemcpy(first.data(),cache.base,Guard*4,cudaMemcpyDeviceToHost));
        Rt(cudaMemcpy(last.data(),(uint8_t*)cache.base+cache.bytes-Guard*4,Guard*4,cudaMemcpyDeviceToHost));
        for(int g=0;g<Guard;g++)if(first[g]!=0x56565656||last[g]!=0x56565656)throw std::runtime_error("input guard corruption");
        HalfGuards(decoded);
        if(s.splits>1) {
            bool meaningful=false;for(size_t i=Guard+D+1;i<Guard+pn;i+=D+2) {
                float x;std::memcpy(&x,&r.partial[i],4);meaningful|=x>0;
                if(s.pattern==4) { float m;std::memcpy(&m,&r.partial[i-1],4); meaningful|=std::isfinite(m); }
            }
            if(!meaningful)throw std::runtime_error("partial fixture did not exercise a nonempty attention sum");
        }
        return r;
    }
};
static void Save(const fs::path& p,const std::vector<uint32_t>& a) {std::ofstream f(p,std::ios::binary);f.write((char*)a.data(),a.size()*4);}
static void Equal(const Result& a,const Result& b,const Shape& s,int variant,const fs::path& output) {
    for(int field=0;field<2;field++) {
        const auto& x=field?a.partial:a.output;const auto& y=field?b.partial:b.output;
        auto it=std::mismatch(x.begin(),x.end(),y.begin());if(it.first==x.end())continue;
        size_t at=it.first-x.begin();Save(output/"failure-reference-output.f32",a.output);Save(output/"failure-actual-output.f32",b.output);
        Save(output/"failure-reference-partial.f32",a.partial);Save(output/"failure-actual-partial.f32",b.partial);
        std::ofstream f(output/"failure.json");f<<"{\"shape\":\""<<Name(s)<<"\",\"variant\":"<<variant<<",\"field\":\""<<(field?"partial":"output")<<"\",\"index_with_guard\":"<<at<<",\"reference_bits\":"<<x[at]<<",\"actual_bits\":"<<y[at]<<"}\n";
        throw std::runtime_error("float-bit mismatch in "+Name(s)+" variant"+std::to_string(variant));
    }
}
static float Time(Fixture& f,Variant& v,int count) {
    cudaEvent_t begin,end;Rt(cudaEventCreate(&begin));Rt(cudaEventCreate(&end));Rt(cudaEventRecord(begin));
    for(int i=0;i<count;i++)f.Issue(v);Rt(cudaEventRecord(end));Rt(cudaEventSynchronize(end));float ms=0;Rt(cudaEventElapsedTime(&ms,begin,end));
    Rt(cudaEventDestroy(end));Rt(cudaEventDestroy(begin));return ms/count;
}
static Result Oracle(Fixture& f,Variant& v,const Shape& s,const fs::path& output) {
    f.Reset();f.Issue(v);auto oracle=f.Read(output);
    for(int repetition=0;repetition<2;repetition++){f.Reset();f.Issue(v);Equal(oracle,f.Read(output),s,0,output);}
    return oracle;
}
static float VerifiedTime(Fixture& f,Variant& v,const Result& oracle,const Shape& s,const fs::path& output) {
    f.Reset();float ms=Time(f,v,3);Equal(oracle,f.Read(output),s,v.id,output);
    if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid complete-pipeline GPU-event interval");return ms;
}
int main(int argc,char** argv) try {
    if(argc<4)throw std::runtime_error("usage: predecoded-attention-bench BUILD_DIR OUT_DIR --resources|--strict|--bench [--long] [--with-no-second-barrier]");
    fs::path build=argv[1],output=argv[2];
    auto relative=fs::weakly_canonical(output).lexically_relative(fs::weakly_canonical(build));
    if(relative.empty()||*relative.begin()=="..")throw std::runtime_error("OUT_DIR must be inside the ignored BUILD_DIR");
    fs::create_directories(output);std::string mode=argv[3];bool longCases=false,noSecondBarrier=false;
    for(int i=4;i<argc;i++) {
        std::string flag=argv[i];if(flag=="--long")longCases=true;else if(flag=="--with-no-second-barrier")noSecondBarrier=true;else throw std::runtime_error("unknown optional flag");
    }
    if(mode!="--resources"&&mode!="--strict"&&mode!="--bench")throw std::runtime_error("unknown mode");
    {std::ofstream f(output/"result.json");f<<"{\"complete\":false,\"hardware_counters\":\"NOT_RUN\"}\n";}
    Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));
    cudaDeviceProp device{};Rt(cudaGetDeviceProperties(&device,0));std::printf("{\"event\":\"device\",\"name\":\"%s\",\"sms\":%d,\"registers_per_sm\":%d,\"shared_per_sm\":%zu}\n",device.name,device.multiProcessorCount,device.regsPerMultiprocessor,device.sharedMemPerMultiprocessor);
    std::ifstream qualified(build/"predecoded-attention-permitted.json");if(!qualified)throw std::runtime_error("missing independent offline build marker; use run.py");
    std::vector<Variant*> variants;struct Cleanup {std::vector<Variant*>& v;~Cleanup(){for(auto p:v)delete p;}}cleanup{variants};
    std::vector<int> deviceRejected;
    for(int id=0;id<(noSecondBarrier?3:2);id++) {
        auto v=new Variant(id,build);if(v->permitted)variants.push_back(v);else{deviceRejected.push_back(id);delete v;}
    }
    if(variants.empty()||variants.front()->id!=0)throw std::runtime_error("missing baseline oracle");
    if(mode=="--resources") {
        std::ofstream f(output/"result.json");f<<"{\"complete\":true,\"validation\":\"resources_only_not_quality\",\"hardware_counters\":\"NOT_RUN\",\"no_second_barrier_requested\":"<<(noSecondBarrier?"true":"false")<<",\"active_variants\":[";
        for(size_t i=0;i<variants.size();i++)f<<(i?",":"")<<variants[i]->id;f<<"],\"device_resource_rejected_variants\":[";
        for(size_t i=0;i<deviceRejected.size();i++)f<<(i?",":"")<<deviceRejected[i];f<<"]}\n";return 0;
    }
    if(variants.size()!=size_t(noSecondBarrier?3:2))throw std::runtime_error("requested candidate failed resource gate; complete candidate quality comparison not performed");
    std::vector<Shape> shapes;
    for(int rows:{1,7,63,64,65,127,128,129,257})for(int splits:{1,2,3,7,8})shapes.push_back({rows%2?0:255,rows,1,1,2,splits,(rows+splits)%4});
    for(int g:{1,2,3,6,8,16,64})for(int pattern:{0,1,3,4})shapes.push_back({257,128/g+1,2,g,2,3,pattern});
    for(int rows:{11,21,22,23,43,65})shapes.push_back({8191,rows,1,6,4,7,3});
    if(longCases)for(int prefix:{131072,250000})shapes.push_back({prefix,129,1,6,4,3,0});
    int strict=0,decodedChecks=0,decodedBoundsChecks=0;
    for(const Shape& s:shapes) {
        Fixture f(s);decodedBoundsChecks+=f.VerifyDecode(*variants[1],output);decodedChecks++;auto oracle=Oracle(f,*variants[0],s,output);
        for(size_t i=1;i<variants.size();i++)for(int repetition=0;repetition<3;repetition++) {f.Reset();f.Issue(*variants[i]);Equal(oracle,f.Read(output),s,variants[i]->id,output);strict++;}
        std::printf("{\"event\":\"strict_shape_pass\",\"shape\":\"%s\",\"variants\":%zu,\"full_output_and_partial_bits\":true}\n",Name(s).c_str(),variants.size());
    }
    if(mode=="--bench") {
        std::vector<Shape> timings={{0,4096,1,6,4,1,0},{19000,4096,1,6,4,1,0},{19000,2657,1,6,4,1,0},{19000,512,2,6,4,3,0}};
        if(longCases) {timings.push_back({131072-4096,4096,1,6,4,1,0});timings.push_back({250000-4096,4096,1,6,4,1,0});}
        for(const Shape& s:timings) {
            Fixture f(s);decodedBoundsChecks+=f.VerifyDecode(*variants[1],output);decodedChecks++;auto oracle=Oracle(f,*variants[0],s,output);
            for(size_t i=1;i<variants.size();i++) {
                auto& v=*variants[i];for(int repetition=0;repetition<3;repetition++){f.Reset();f.Issue(v);Equal(oracle,f.Read(output),s,v.id,output);} // Gate the exact timed shape too.
                for(int repetition=0;repetition<5;repetition++) {
                    float a=VerifiedTime(f,*variants[0],oracle,s,output),b=VerifiedTime(f,v,oracle,s,output),c=VerifiedTime(f,v,oracle,s,output),d=VerifiedTime(f,*variants[0],oracle,s,output);
                    std::printf("{\"event\":\"abba_gpu_ms\",\"shape\":\"%s\",\"total_keys\":%d,\"variant\":%d,\"repetition\":%d,\"calls_per_half\":3,\"a\":%.7g,\"b\":%.7g,\"c\":%.7g,\"d\":%.7g,\"workspace_bytes\":%zu,\"candidate_includes_predecode_attention_combine_every_call\":true,\"poison_and_verify_each_half\":true}\n",Name(s).c_str(),s.prefix+s.rows,v.id,repetition,a,b,c,d,f.HalfCount(256)*2);
                }
            }
        }
    }
    std::ofstream f(output/"result.json");f<<"{\"complete\":true,\"strict_shapes\":"<<shapes.size()<<",\"decoded_fixture_checks\":"<<decodedChecks<<",\"decoded_boundary_checks\":"<<decodedBoundsChecks<<",\"strict_candidate_comparisons\":"<<strict<<",\"long_cases\":"<<(longCases?"true":"false")<<",\"timed\":"<<(mode=="--bench"?"true":"false")<<",\"no_second_barrier_requested\":"<<(noSecondBarrier?"true":"false")<<",\"hardware_counters\":\"NOT_RUN\",\"active_variants\":[";
    for(size_t i=0;i<variants.size();i++)f<<(i?",":"")<<variants[i]->id;f<<"],\"device_resource_rejected_variants\":[";
    for(size_t i=0;i<deviceRejected.size();i++)f<<(i?",":"")<<deviceRejected[i];f<<"]}\n";
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"attention-bench failed: %s\n",e.what());return 1;}
