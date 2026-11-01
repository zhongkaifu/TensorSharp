// Standalone validation only: dynamically loads isolated experiment cubins.
// --bench always runs complete bitwise gates before event timing. No production integration.
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
    int id,shared,regs=0,local=0,stat=0,active=0;bool permitted=true;CUmodule module=nullptr;CUfunction attention=nullptr,combine=nullptr;
    Variant(int v,const fs::path& dir):id(v),shared(v==4?50176:(v==0||v==2?68608:51200)) {
        Cu(cuModuleLoad(&module,(dir/("variant-"+std::to_string(v)+".cubin")).c_str()));
        Cu(cuModuleGetFunction(&attention,module,"ts_bench_attn_prefill"));
        Cu(cuModuleGetFunction(&combine,module,"ts_bench_attn_combine"));
        Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,attention));
        Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,attention));
        Cu(cuFuncGetAttribute(&stat,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,attention));
        Cu(cuFuncSetAttribute(attention,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
        Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,attention,256,shared));
        permitted=v<2 || (regs<=128 && shared+stat<=51200 && local==0 && active>=2);
        std::printf("{\"event\":\"device_resources\",\"variant\":%d,\"regs\":%d,\"local_bytes\":%d,\"dynamic_shared\":%d,\"static_shared\":%d,\"potential_active_ctas\":%d}\n",id,regs,local,shared,stat,active);
        if(!permitted)std::printf("{\"event\":\"resource_rejected_no_gpu_run\",\"variant\":%d}\n",id);
    }
    ~Variant() { if(module) cuModuleUnload(module); }
};
struct Result { std::vector<uint32_t> output,partial; };
static void Save(const fs::path& p,const std::vector<uint32_t>& a);
struct Fixture {
    Shape s;int heads,keys,blocks;size_t rowHeads,qn,pn,blockBytes;long long ko=0,vo;
    Allocation cache,q,gate,ends,table,out,partial;
    Fixture(Shape shape):s(shape),heads(s.g*s.kv),keys(s.prefix+s.rows),blocks((keys+B-1)/B),rowHeads((size_t)s.seq*s.rows*heads),
        qn(rowHeads*D),pn(rowHeads*s.splits*(D+2)),blockBytes(2ull*s.kv*B*Q8),vo((long long)s.kv*B*Q8),
        cache((size_t)s.seq*blocks*blockBytes+2*Guard*4),q(qn*4),gate(qn*4),ends((size_t)s.seq*s.rows*4),
        table((size_t)s.seq*blocks*sizeof(void*)),out((qn+2*Guard)*4),partial((pn+2*Guard)*4) {
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
        Rt(cudaMemcpy(cache.base,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        Rt(cudaMemcpy(q.base,hostQ.data(),qn*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(gate.base,hostGate.data(),qn*4,cudaMemcpyHostToDevice));
        Rt(cudaMemcpy(ends.base,hostEnds.data(),hostEnds.size()*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(table.base,pointers.data(),pointers.size()*sizeof(void*),cudaMemcpyHostToDevice));
    }
    void Reset() {
        std::vector<uint32_t> a(qn+2*Guard,Canary),b(pn+2*Guard,Canary);
        Rt(cudaMemcpy(out.base,a.data(),a.size()*4,cudaMemcpyHostToDevice));Rt(cudaMemcpy(partial.base,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    }
    void Issue(Variant& v) {
        void *qp=q.base,*gp=gate.base,*ep=ends.base,*tp=table.base,*pp=(uint32_t*)partial.base+Guard,*op=(uint32_t*)out.base+Guard;
        int maxBlocks=blocks,rows=s.rows,nh=heads,nk=s.kv,sp=s.splits;float scale=1/std::sqrt((float)D);long long koff=ko,voff=vo;
        void* args[]={&qp,&gp,&ep,&tp,&maxBlocks,&koff,&voff,&rows,&nh,&nk,&scale,&sp,&pp,&op};
        unsigned tiles=(rows*s.g+127)/128,physical=tiles*sp*(v.id>=2?2:1);
        Cu(cuLaunchKernel(v.attention,physical,nk,s.seq,256,1,1,v.shared,nullptr,args,nullptr));
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
    f.Reset();float ms=Time(f,v,3);Equal(oracle,f.Read(output),s,v.id,output);return ms;
}
int main(int argc,char** argv) try {
    if(argc<4)throw std::runtime_error("usage: attention-bench BUILD_DIR OUT_DIR --resources|--strict|--bench [--long]");
    fs::path build=argv[1],output=argv[2];
    auto relative=fs::weakly_canonical(output).lexically_relative(fs::weakly_canonical(build));
    if(relative.empty()||*relative.begin()=="..")throw std::runtime_error("OUT_DIR must be inside the ignored BUILD_DIR");
    fs::create_directories(output);std::string mode=argv[3];bool longCases=argc>4&&std::string(argv[4])=="--long";
    if(mode!="--resources"&&mode!="--strict"&&mode!="--bench")throw std::runtime_error("unknown mode");
    {std::ofstream f(output/"run.json");f<<"{\"complete\":false,\"hardware_counters\":\"NOT_RUN\"}\n";}
    Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));
    cudaDeviceProp device{};Rt(cudaGetDeviceProperties(&device,0));std::printf("{\"event\":\"device\",\"name\":\"%s\",\"sms\":%d,\"registers_per_sm\":%d,\"shared_per_sm\":%zu}\n",device.name,device.multiProcessorCount,device.regsPerMultiprocessor,device.sharedMemPerMultiprocessor);
    std::ifstream qualified(build/"qualified-variants.txt");if(!qualified)throw std::runtime_error("missing offline resource gates");
    std::vector<Variant*> variants;struct Cleanup {std::vector<Variant*>& v;~Cleanup(){for(auto p:v)delete p;}}cleanup{variants};
    std::vector<int> deviceRejected;
    int id,last=-1;while(qualified>>id) {
        if(id<0||id>4||id<=last)throw std::runtime_error("malformed/duplicate qualification marker");last=id;
        auto v=new Variant(id,build);if(v->permitted)variants.push_back(v);else{deviceRejected.push_back(id);delete v;}
    }
    if(!qualified.eof())throw std::runtime_error("malformed qualification marker");
    if(variants.empty()||variants.front()->id!=0)throw std::runtime_error("missing baseline oracle");
    if(mode=="--resources") {
        std::ofstream f(output/"run.json");f<<"{\"complete\":true,\"validation\":\"resources_only_not_quality\",\"hardware_counters\":\"NOT_RUN\",\"device_resource_rejected_variants\":[";
        for(size_t i=0;i<deviceRejected.size();i++)f<<(i?",":"")<<deviceRejected[i];f<<"]}\n";return 0;
    }
    if(variants.size()<2)throw std::runtime_error("no candidate passed resource gates; no candidate quality comparison performed");
    std::vector<Shape> shapes;
    for(int rows:{1,7,63,64,65,127,128,129,257})for(int splits:{1,2,3,7,8})shapes.push_back({rows%2?0:255,rows,1,1,2,splits,(rows+splits)%4});
    for(int g:{1,2,3,6,8,16,64})for(int pattern:{0,1,3,4})shapes.push_back({257,128/g+1,2,g,2,3,pattern});
    for(int rows:{11,21,22,23,43,65})shapes.push_back({8191,rows,1,6,4,7,3});
    if(longCases)for(int prefix:{131072,250000})shapes.push_back({prefix,129,1,6,4,3,0});
    int strict=0;
    for(const Shape& s:shapes) {
        Fixture f(s);auto oracle=Oracle(f,*variants[0],s,output);
        for(size_t i=1;i<variants.size();i++)for(int repetition=0;repetition<3;repetition++) {f.Reset();f.Issue(*variants[i]);Equal(oracle,f.Read(output),s,variants[i]->id,output);strict++;}
        std::printf("{\"event\":\"strict_shape_pass\",\"shape\":\"%s\",\"variants\":%zu,\"full_output_and_partial_bits\":true}\n",Name(s).c_str(),variants.size());
    }
    if(mode=="--bench") {
        std::vector<Shape> timings={{0,4096,1,6,4,1,0},{19000,4096,1,6,4,1,0},{19000,2657,1,6,4,1,0},{19000,512,2,6,4,3,0}};
        if(longCases) {timings.push_back({131072-4096,4096,1,6,4,1,0});timings.push_back({250000-4096,4096,1,6,4,1,0});}
        for(const Shape& s:timings) {
            Fixture f(s);auto oracle=Oracle(f,*variants[0],s,output);
            for(size_t i=1;i<variants.size();i++) {
                auto& v=*variants[i];for(int repetition=0;repetition<3;repetition++){f.Reset();f.Issue(v);Equal(oracle,f.Read(output),s,v.id,output);} // Gate the exact timed shape too.
                for(int repetition=0;repetition<5;repetition++) {
                    float a=VerifiedTime(f,*variants[0],oracle,s,output),b=VerifiedTime(f,v,oracle,s,output),c=VerifiedTime(f,v,oracle,s,output),d=VerifiedTime(f,*variants[0],oracle,s,output);
                    std::printf("{\"event\":\"abba_gpu_ms\",\"shape\":\"%s\",\"total_keys\":%d,\"variant\":%d,\"repetition\":%d,\"kernels_per_half\":3,\"a\":%.7g,\"b\":%.7g,\"c\":%.7g,\"d\":%.7g,\"complete_subtiles_and_combine\":true,\"poison_and_verify_each_half\":true}\n",Name(s).c_str(),s.prefix+s.rows,v.id,repetition,a,b,c,d);
                }
            }
        }
    }
    std::ofstream f(output/"run.json");f<<"{\"complete\":true,\"strict_shapes\":"<<shapes.size()<<",\"strict_candidate_comparisons\":"<<strict<<",\"long_cases\":"<<(longCases?"true":"false")<<",\"timed\":"<<(mode=="--bench"?"true":"false")<<",\"hardware_counters\":\"NOT_RUN\",\"active_variants\":[";
    for(size_t i=0;i<variants.size();i++)f<<(i?",":"")<<variants[i]->id;f<<"],\"device_resource_rejected_variants\":[";
    for(size_t i=0;i<deviceRejected.size();i++)f<<(i?",":"")<<deviceRejected[i];f<<"]}\n";
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"attention-bench failed: %s\n",e.what());return 1;}
