// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// DLL CPU/CUDA graph: same batch arithmetic, compact file expert axis.
// TSG_PUBLIC_API_ONLY builds an executable that imports no test-hook entry points,
// for validating a clean BUILD_TESTS=OFF GgmlOps DLL without re-enabling hooks.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>
#include <limits>
#include <string>
#include "ggml.h"
#include "../ggml_ops_moe_file_pipeline_plan.h"
#include "../ggml_ops_moe_file_pipeline_config.h"

extern "C" {
const char* TSGgml_GetLastError();
int TSGgml_IsBackendAvailable(int);
void TSGgml_Shutdown();
void TSGgml_InvalidateHostBuffer(void*);
void TSGgml_ReleaseReuseComputeBuffers();
void TSGgml_SetHostMoeThreads(int);
int TSGgml_AttachSharedCacheBudgetWithHost(void*, std::uint64_t(*)(void*,int,int,std::int64_t),int(*)(void*,std::uint64_t),void(*)(void*,std::uint64_t),int);
int TSGgml_DetachSharedCacheBudget(void*);
int TSGgml_BeginHostMoeExpertCacheScope(std::int64_t,int,int,std::uint64_t*);
int TSGgml_TrimHostMoeExpertCacheScope(std::uint64_t,std::int64_t,std::int64_t*);
int TSGgml_EndHostMoeExpertCacheScope(std::uint64_t,std::int64_t*);
int TSGgml_BeginHostMoeFilePipelineScope(int,int,std::int64_t,std::int64_t,std::uint64_t*);
int TSGgml_EndHostMoeFilePipelineScope(std::uint64_t);
int TSGgml_HostMoeFilePipelineStats(std::int64_t*,std::int64_t*,std::int64_t*);
int TSGgml_RegisterHostFileSource(const void*,std::int64_t,const char*,std::int64_t);
int TSGgml_TestHostMoeFileSegment(float*,float*,int,int,int,int,int,const std::int32_t*,const float*,
    void*,int,std::int64_t,void*,int,std::int64_t,void*,int,std::int64_t);
int TSGgml_TestHostMoeFileCpuGraph(float*,float*,int,int,int,int,int,const std::int32_t*,const float*,
    void*,int,std::int64_t,void*,int,std::int64_t,void*,int,std::int64_t);
void TSGgml_TestHostMoeFileReadFailNext();
void TSGgml_TestHostMoeFileStreamFailNext(int);
std::uint64_t TSGgml_TestHostMoeFileStreamDrains();
int TSGgml_MoEFFNPrefillSwiGLUQuantF32(float*,float*,int,int,int,int,int,const std::int32_t*,const float*,
    void*,int,std::int64_t,std::int64_t,std::int64_t,void*,int,std::int64_t,std::int64_t,std::int64_t,
    void*,int,std::int64_t,std::int64_t,std::int64_t,const float*,const float*,const float*,int,float,float,int);
}
namespace {
int checks=0;
void require(bool v,const char* s) { ++checks; if(!v) throw std::runtime_error(s); }
void env(const char* k,const char* v) {
#ifdef _WIN32
_putenv_s(k,v);
#else
setenv(k,v,1);
#endif
}
struct Ledger {
    struct Ticket { std::int64_t bytes; int kind; };
    std::map<std::uint64_t,Ticket> tickets;
    std::int64_t cap=64LL<<20,live=0,peak=0;
    std::int64_t device_cap=64LL<<20,device_live=0,host_live=0,device_peak=0,host_peak=0;
    std::uint64_t next=0,calls=0;
    std::uint64_t device_refusals=0,device_calls=0;
    std::uint64_t deny_call=0;
    bool valid=true,fail_commit=false,fail_host_commit=false,graphs=false;
    static std::uint64_t reserve(void* p,int rank,int kind,std::int64_t bytes) {
        auto& l=*static_cast<Ledger*>(p); ++l.calls;
        if(l.calls==l.deny_call) return 0;
        if(rank!=0 || (kind!=3 && !(l.graphs && kind==2)) || bytes<=0) { l.valid=false; return 0; }
        if(kind==2) ++l.device_calls;
        auto& live=kind==2?l.device_live:l.host_live;
        if(bytes>(kind==2?l.device_cap:l.cap)-live) { if(kind==2) ++l.device_refusals; return 0; }
        auto id=++l.next; l.tickets.emplace(id,Ticket{bytes,kind}); live+=bytes; l.live+=bytes;
        l.peak=std::max(l.peak,l.live); l.device_peak=std::max(l.device_peak,l.device_live); l.host_peak=std::max(l.host_peak,l.host_live); return id;
    }
    static int commit(void* p,std::uint64_t id) {
        auto& l=*static_cast<Ledger*>(p); const auto found=l.tickets.find(id);
        return !l.fail_commit && found!=l.tickets.end() && !(l.fail_host_commit && found->second.kind==3);
    }
    static void release(void* p,std::uint64_t id) {
        auto& l=*static_cast<Ledger*>(p); auto i=l.tickets.find(id);
        if(i==l.tickets.end()) { l.valid=false; return; }
        l.live-=i->second.bytes; (i->second.kind==2?l.device_live:l.host_live)-=i->second.bytes; l.tickets.erase(i);
    }
};
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
constexpr int H=512,E=7,K=3,Q8=8;
#else
constexpr int H=256,E=7,K=3,Q8=8;
#endif
std::vector<std::uint8_t> weight(int seed) {
    std::vector<std::uint8_t> w(std::size_t(H)*H*E/32*34);
    for(std::size_t b=0;b<w.size()/34;++b) { w[b*34]=0; w[b*34+1]=0x20;
        for(int j=0;j<32;++j) w[b*34+2+j]=static_cast<std::uint8_t>(static_cast<std::int8_t>((int(b%23)+j*3+seed)%11-5)); }
    return w;
}
std::vector<std::uint8_t> mixed_weight(ggml_type type,int seed) {
    ggml_quantize_init(type);
    const auto stride=ggml_row_size(type,H)*H;
    std::vector<std::uint8_t> w(stride*E);
    std::vector<float> x(H*H),importance(H,1.f);
    for(int e=0;e<E;++e) {
        for(std::size_t i=0;i<x.size();++i) x[i]=(int((i*17+e*31+seed)%43)-21)/256.f;
        require(ggml_quantize_chunk(type,x.data(),w.data()+stride*e,0,H,H,
            ggml_quantize_requires_imatrix(type)?importance.data():nullptr)==stride,"quantized expert length");
    }
    return w;
}
}
int main(int argc,char** argv) {
    bool cuda=false,mixed=false;
    int threshold_override=-1,pipeline_override=-1;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--cuda")==0) cuda=true;
        else if(std::strcmp(argv[i],"--mixed")==0) mixed=true;
        else if(std::strcmp(argv[i],"--pipeline-scope")==0 && i+1<argc
            && (std::strcmp(argv[i+1],"0")==0 || std::strcmp(argv[i+1],"1")==0)) pipeline_override=std::atoi(argv[++i]);
        else if(std::strcmp(argv[i],"--threshold-override")==0 && i+1<argc
            && (std::strcmp(argv[i+1],"0")==0 || std::strcmp(argv[i+1],"128")==0)) threshold_override=std::atoi(argv[++i]);
        else { std::fprintf(stderr,"Unknown argument\n"); return 1; }
    }
    env("TS_GGML_BACKEND",cuda?"cuda":"cpu");
    if(threshold_override>=0) env("TS_HOST_MOE_DEVICE_MIN_BATCH",threshold_override==0?"0":"128");
    else if(cuda) {
#ifdef _WIN32
        _putenv_s("TS_HOST_MOE_DEVICE_MIN_BATCH", "");
#else
        unsetenv("TS_HOST_MOE_DEVICE_MIN_BATCH");
#endif
    } else env("TS_HOST_MOE_DEVICE_MIN_BATCH","0");
    env("TS_HOST_MOE_EXPERT_CACHE_MB","0");
    env("TS_HOST_MOE_PIN","0"); env("TS_CPU_MOE_THREADS","2");
    Ledger ledger; std::uint64_t scope=0,pipeline_scope=0; bool attached=false;
    auto dir=std::filesystem::temp_directory_path()/std::filesystem::u8path(u8"tensorsharp-专家-file-batch");
    int result=0;
    try {
        require(TSGgml_IsBackendAvailable(cuda?3:2)==1,"required backend unavailable"); TSGgml_SetHostMoeThreads(2);
        if(pipeline_override>=0)
            require(TSGgml_BeginHostMoeFilePipelineScope(pipeline_override,64,1,16u<<20,&pipeline_scope)==1
                && pipeline_scope!=0,"pipeline scope begin failed");
        std::filesystem::create_directories(dir);
        const int types[]={Q8,mixed?int(GGML_TYPE_IQ1_M):Q8,mixed?int(GGML_TYPE_Q4_0):Q8};
        auto gate=weight(1),up=mixed?mixed_weight(GGML_TYPE_IQ1_M,2):weight(2),down=mixed?mixed_weight(GGML_TYPE_Q4_0,3):weight(3);
        std::vector<std::uint8_t>* all[]={&gate,&up,&down};
        for(int p=0;p<3;++p) { auto file=dir/(std::to_string(p)+".bin");
            std::ofstream out(file,std::ios::binary); out.write("PAD",3); out.write(reinterpret_cast<const char*>(all[p]->data()),all[p]->size()); out.close();
            require(TSGgml_RegisterHostFileSource(all[p]->data(),all[p]->size(),file.u8string().c_str(),3)==1,"source registration failed"); }
        auto invalid=dir/"0.bin";
        require(TSGgml_RegisterHostFileSource(&ledger,gate.size()+1,invalid.u8string().c_str(),3)==0,"source extent beyond EOF accepted");
        const std::vector<int> shapes=threshold_override>=0 ? std::vector<int>{33,129}
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
            // Mixed IQ1/Q4 payloads are smaller; the all-seven-expert case
            // exceeds the 1MiB read-job boundary without enlarging the fixture.
            : mixed ? std::vector<int>{257} : std::vector<int>{128,257}
#else
            : cuda ? std::vector<int>{32,33,37,77,100,127,128,129,257,512,513}
            : std::vector<int>{1,8,31,129,257,512,513}
#endif
            ;
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
        require(cuda && pipeline_override==1,"large parallel-read fixture requires explicit CUDA pipeline scope");
#endif
        for(int rows : shapes) {
            if(cuda && rows<=8) continue;
            std::vector<float> x(rows*H),ref(rows*H),out(rows*H),routes(rows*K);
            std::vector<std::int32_t> ids(rows*K);
            for(std::size_t i=0;i<x.size();++i) x[i]=(int(i%31)-15)/16.f;
            // N257 touches every expert in unsorted route order: all three
            // projections now merge to one read each. Other batches retain
            // noncontiguous runs; N513 retains the repeated-ID U<K boundary.
            for(int r=0;r<rows;++r) for(int k=0;k<K;++k) {
                ids[r*K+k]=cuda ? (rows==257?(E-1-r%E+k)%E:(r%2?2:0)+k*2)
                    : rows==513?2:rows==257?(E-1-r% E+k)%E:(r%2?2:5)+(k==2?1:0);
                routes[r*K+k]=(k+1)/6.f;
            }
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
            // Independent expected geometry; the runner also requires native
            // diagnostic read_jobs>chunks, proving that the actual v11 path
            // selects multiple bounded source reads per DMA ring slot.
            std::vector<std::int32_t> selected=ids;
            std::sort(selected.begin(),selected.end());
            selected.erase(std::unique(selected.begin(),selected.end()),selected.end());
            std::array<std::size_t,3> expected_bytes{};
            std::size_t payload=0;
            for(int part=0;part<3;++part) {
                expected_bytes[part]=all[part]->size()/E*selected.size();
                payload+=(expected_bytes[part]+63)&~std::size_t(63);
            }
            const auto slot=tsg::host_moe_file_pipeline_slot_bytes(payload,16u<<20);
            tsg::HostMoeFilePipelinePlan plan(slot);
            for(int part=0;part<3;++part)
                for(std::size_t i=0;i<selected.size();++i) {
                    const auto stride=all[part]->size()/E;
                    plan.append(part,selected[i]*stride,i*stride,stride);
                }
            plan.validate(expected_bytes);
            std::size_t jobs=0;
            for(std::size_t i=0;i<plan.transfers().size();++i) jobs+=plan.read_job_count(i,1u<<20);
            require(slot>1u<<20 && jobs>plan.transfers().size(),"large fixture does not exercise intra-slot parallel reads");
            std::printf("PARALLEL-READ-GEOMETRY rows=%d hidden=%d experts=%d source_bytes=%zu slot_bytes=%zu transfers=%zu read_jobs=%zu\n",
                rows,H,E,payload,slot,plan.transfers().size(),jobs);
#endif
            auto run=[&](std::vector<float>& y) {
#if !defined(TSG_PUBLIC_API_ONLY)
                if(rows<=8) return TSGgml_TestHostMoeFileSegment(x.data(),y.data(),rows,H,H,E,K,ids.data(),routes.data(),
                    gate.data(),types[0],gate.size(),up.data(),types[1],up.size(),down.data(),types[2],down.size());
#endif
                return TSGgml_MoEFFNPrefillSwiGLUQuantF32(x.data(),y.data(),rows,H,H,E,K,ids.data(),routes.data(),
                gate.data(),types[0],H,H,gate.size(),up.data(),types[1],H,H,up.size(),down.data(),types[2],H,H,down.size(),nullptr,nullptr,nullptr,0,1.702f,7.f,1); };
            const bool expect_stream=cuda && (threshold_override<0 || (threshold_override>0 && rows>=threshold_override));
            if(expect_stream && rows<128 && threshold_override<0) {
                // Explicit ordinary device execution is the independent oracle
                // below the unchanged ordinary host-offload threshold. Use
                // separate pointers and invalidate them before attaching quota.
                auto a=gate,b=up,c=down;
                require(TSGgml_MoEFFNPrefillSwiGLUQuantF32(x.data(),ref.data(),rows,H,H,E,K,ids.data(),routes.data(),
                    a.data(),types[0],H,H,a.size(),b.data(),types[1],H,H,b.size(),c.data(),types[2],H,H,c.size(),
                    nullptr,nullptr,nullptr,0,1.702f,7.f,0)==1,"ordinary device reference failed");
                TSGgml_InvalidateHostBuffer(a.data()); TSGgml_InvalidateHostBuffer(b.data()); TSGgml_InvalidateHostBuffer(c.data());
            } else require(run(ref)==1,"ordinary backend reference failed");
            if(cuda) TSGgml_ReleaseReuseComputeBuffers(); // No pre-existing unaccounted graph owner at attach.
            ledger.graphs=cuda;
            require(TSGgml_AttachSharedCacheBudgetWithHost(&ledger,Ledger::reserve,Ledger::commit,Ledger::release,cuda?1:0)==1,"budget attach failed"); attached=true;
            require(TSGgml_BeginHostMoeExpertCacheScope(0,128,1,&scope)==1,"scope begin failed");
            if(cuda && !expect_stream) {
                const auto device_calls=ledger.device_calls;
                require(run(out)==1 && ledger.host_live>0 && ledger.device_live==0 && ledger.device_calls==device_calls,
                    "explicit stream threshold or disable was ignored");
                require(std::memcmp(out.data(),ref.data(),out.size()*sizeof(float))==0,"explicit threshold CPU result changed");
                std::int64_t freed=0;
                require(TSGgml_EndHostMoeExpertCacheScope(scope,&freed)==1 && ledger.live==0,"threshold scope leaked");scope=0;
                require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"threshold detach failed");attached=false;
                continue;
            }
            if(cuda) {
                const auto trim=[&] { std::int64_t freed=0; require(TSGgml_TrimHostMoeExpertCacheScope(scope,0,&freed)==1 && ledger.live==0,"stream trim retained owners"); };
                const auto unchanged=[&] { require(std::all_of(out.begin(),out.end(),[](float f){return f==-12345.f;}),"failed stream published output"); };
                const auto compare=[&](const std::vector<float>& expected) {
                    double error=0,norm=0,max_abs=0;
                    for(std::size_t i=0;i<out.size();++i) {
                        require(std::isfinite(out[i]) && std::isfinite(expected[i]),"nonfinite CUDA result");
                        const double delta=double(out[i])-expected[i]; error+=delta*delta; norm+=double(expected[i])*expected[i]; max_abs=std::max(max_abs,std::abs(delta));
                    }
                    const double relative=std::sqrt(error/std::max(norm,1e-30));
                    std::printf("N%d %s ordinary-CUDA compact comparison relL2=%.9g maxAbs=%.9g bitexact=%d\n",rows,mixed?"mixed":"Q8",relative,max_abs,std::memcmp(out.data(),expected.data(),out.size()*sizeof(float))==0);
                    require(relative<=1e-4 && max_abs<=1e-4,"compact CUDA differs from ordinary CUDA oracle");
                };
                ledger.device_cap=0; const auto refused=ledger.device_refusals;
                require(run(out)==1 && ledger.device_live==0 && ledger.host_live>0 && ledger.device_refusals>refused,"zero device quota did not use bounded CPU fallback");
                const auto cpu_reference=out;
                trim();
                require(run(out)==1 && std::memcmp(out.data(),cpu_reference.data(),out.size()*sizeof(float))==0,"repeated refused-device CPU result changed");
                trim(); ledger.device_cap=64LL<<20;
                ledger.cap=1; std::fill(out.begin(),out.end(),-12345.f);
                require(run(out)==0 && ledger.live==0,"host refusal leaked CUDA arena or bypassed quota"); unchanged(); ledger.cap=64LL<<20;
                ledger.fail_host_commit=true;
                require(run(out)==0 && ledger.live==0,"host payload commit failure leaked CUDA or pinned ownership");
                unchanged(); ledger.fail_host_commit=false;
#if !defined(TSG_PUBLIC_API_ONLY)
                TSGgml_TestHostMoeFileReadFailNext();
                require(run(out)==0 && ledger.live==0,"failed file read leaked stream graph"); unchanged();
                const char* pipeline=std::getenv("TS_HOST_MOE_FILE_PIPELINE");
                const int last_stage=(pipeline_override>=0 ? pipeline_override==1 && rows>=64
                    : pipeline && std::strcmp(pipeline,"1")==0)?3:2;
                for(int stage=1;stage<=last_stage;++stage) {
                    const auto drains=TSGgml_TestHostMoeFileStreamDrains();
                    TSGgml_TestHostMoeFileStreamFailNext(stage);
                    require(run(out)==0 && ledger.live==0,"partial upload/compute silently retried or leaked");
                    require(TSGgml_TestHostMoeFileStreamDrains()>drains,"failed stream did not drain before returning"); unchanged();
                }
#endif
                auto original_gate=gate; std::fill(gate.begin(),gate.end(),0);
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
                // The ordinary oracle above has a different physical expert
                // axis, which can select different vendor arithmetic. Compare
                // the pipeline additionally against exactly the same compact
                // graph and registered source bytes with its scope disabled.
                std::vector<float> parallel_control(out.size());
                require(TSGgml_EndHostMoeFilePipelineScope(pipeline_scope)==1,
                    "parallel control failed to retire previous pipeline storage"); pipeline_scope=0; trim();
                require(TSGgml_BeginHostMoeFilePipelineScope(0,64,1,16u<<20,&pipeline_scope)==1,
                    "parallel control disabled scope failed");
                require(run(parallel_control)==1,"identical compact disabled-pipeline control failed");
                require(TSGgml_EndHostMoeFilePipelineScope(pipeline_scope)==1,
                    "parallel control failed to retire its ring"); pipeline_scope=0; trim();
                require(TSGgml_BeginHostMoeFilePipelineScope(1,64,1,16u<<20,&pipeline_scope)==1,
                    "parallel control enabled scope failed");
#endif
                require(run(out)==1 && ledger.device_live>0 && ledger.host_live>0 && ledger.valid,"compact CUDA stream failed or uncharged"); compare(ref);
#if defined(TSG_TEST_HOST_MOE_FILE_BATCH_LARGE)
                require(std::memcmp(out.data(),parallel_control.data(),out.size()*sizeof(float))==0,
                    "parallel reads changed the identical compact graph arithmetic");
                std::printf("PARALLEL-COMPACT-EXACT rows=%d floats=%zu bitexact=1\n",rows,out.size());
#endif
                if(pipeline_override>=0) {
                    std::int64_t calls=0,bytes=0,ring=0;
                    require(TSGgml_HostMoeFilePipelineStats(&calls,&bytes,&ring)==1,"pipeline stats failed");
                    require((pipeline_override==1 && rows>=64) ? ring>0 && ring<=16u<<20 && calls>0 && bytes>0
                        : ring==0,"scoped pipeline threshold or disable was ignored");
                }
                const auto gpu_reference=out;
                const auto calls=ledger.calls;
                const auto live=ledger.live;
                require(run(out)==1 && ledger.calls==calls && ledger.live==live,"second layer did not reuse host/device arenas");
                require(std::memcmp(out.data(),gpu_reference.data(),out.size()*sizeof(float))==0,"retained stream changed output");
                trim(); require(run(out)==1,"stream failed after trim");
                require(std::memcmp(out.data(),gpu_reference.data(),out.size()*sizeof(float))==0,"recreated stream changed output");
                std::copy(original_gate.begin(),original_gate.end(),gate.begin()); trim();
                std::int64_t freed=0; require(TSGgml_EndHostMoeExpertCacheScope(scope,&freed)==1 && ledger.live==0,"stream scope cleanup leaked"); scope=0;
                require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"stream scope did not detach"); attached=false;
                ledger.graphs=false;
                require(TSGgml_AttachSharedCacheBudgetWithHost(&ledger,Ledger::reserve,Ledger::commit,Ledger::release,0)==1,"host-only budget attach failed"); attached=true;
                require(TSGgml_BeginHostMoeExpertCacheScope(0,128,1,&scope)==1,"host-only scope begin failed");
                require(run(out)==1 && ledger.device_live==0 && ledger.host_live>0 && ledger.valid
                    && std::memcmp(out.data(),cpu_reference.data(),out.size()*sizeof(float))==0,"host-only scope created an unaccounted CUDA stream");
                require(TSGgml_EndHostMoeExpertCacheScope(scope,&freed)==1 && ledger.live==0,"host-only scope cleanup leaked"); scope=0;
                require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"host-only scope did not detach"); attached=false;
#if !defined(TSG_PUBLIC_API_ONLY)
                // Ordinary CUDA's ID compaction requires unique top-k entries.
                // Validate repeated routes against the independent full CPU graph,
                // including both U>=K and U<K, while CUDA has ample quota.
                if(rows==129 || rows==513) {
                    for(int r=0;r<rows;++r) for(int k=0;k<K;++k)
                        ids[r*K+k]=rows==513?2:(r%2?2:5)+(k==2?1:0);
                    require(TSGgml_TestHostMoeFileCpuGraph(x.data(),ref.data(),rows,H,H,E,K,ids.data(),routes.data(),
                        gate.data(),types[0],gate.size(),up.data(),types[1],up.size(),down.data(),types[2],down.size())==1,
                        "uncompacted duplicate CPU oracle failed");
                    ledger.graphs=true;
                    require(TSGgml_AttachSharedCacheBudgetWithHost(&ledger,Ledger::reserve,Ledger::commit,Ledger::release,1)==1,"duplicate budget attach failed"); attached=true;
                    require(TSGgml_BeginHostMoeExpertCacheScope(0,128,1,&scope)==1,"duplicate scope begin failed");
                    const auto device_calls=ledger.device_calls;
                    const auto drains=TSGgml_TestHostMoeFileStreamDrains();
                    std::fill(gate.begin(),gate.end(),0);
                    require(run(out)==1 && ledger.host_live>0 && ledger.device_live==0
                        && ledger.device_calls==device_calls && TSGgml_TestHostMoeFileStreamDrains()==drains,
                        "duplicate routes entered CUDA instead of bounded CPU");
                    require(std::memcmp(out.data(),ref.data(),out.size()*sizeof(float))==0,"duplicate bounded CPU differs from full CPU oracle");
                    std::copy(original_gate.begin(),original_gate.end(),gate.begin());
                    require(TSGgml_EndHostMoeExpertCacheScope(scope,&freed)==1 && ledger.live==0,"duplicate scope retained owners"); scope=0;
                    require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"duplicate scope did not detach"); attached=false;
                }
#else
                if(rows==129 || rows==513) {
                    for(int r=0;r<rows;++r) for(int k=0;k<K;++k)
                        ids[r*K+k]=rows==513?2:(r%2?2:5)+(k==2?1:0);
                    // Clean DLL has no force-CPU test entry point. First capture
                    // the public host-only budget arm, then enable graph quota;
                    // repeated routes must still use exactly that CPU result.
                    for(int graph_flag : {0,1}) {
                        ledger.graphs=graph_flag!=0;
                        require(TSGgml_AttachSharedCacheBudgetWithHost(&ledger,Ledger::reserve,Ledger::commit,Ledger::release,graph_flag)==1,"public duplicate attach failed"); attached=true;
                        require(TSGgml_BeginHostMoeExpertCacheScope(0,128,1,&scope)==1,"public duplicate scope failed");
                        const auto device_calls=ledger.device_calls;
                        std::fill(gate.begin(),gate.end(),0);
                        require(run(out)==1 && ledger.host_live>0 && ledger.device_live==0 && ledger.device_calls==device_calls,
                            "public duplicate route entered CUDA");
                        if(graph_flag==0) ref=out;
                        else require(std::memcmp(out.data(),ref.data(),out.size()*sizeof(float))==0,"public duplicate result changed with graph budget");
                        std::copy(original_gate.begin(),original_gate.end(),gate.begin());
                        require(TSGgml_EndHostMoeExpertCacheScope(scope,&freed)==1 && ledger.live==0,"public duplicate owners leaked"); scope=0;
                        require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"public duplicate detach failed"); attached=false;
                    }
                }
#endif
                continue;
            }
            ledger.cap=1; std::fill(out.begin(),out.end(),-12345.f);
            require(run(out)==0 && ledger.live==0,"quota bypass or leak");
            require(std::all_of(out.begin(),out.end(),[](float f){return f==-12345.f;}),"refused graph changed output");
            ledger.cap=64LL<<20; ledger.fail_commit=true;
            require(run(out)==0 && ledger.live==0,"commit failure leaked staging"); ledger.fail_commit=false;
#if defined(TSG_PUBLIC_API_ONLY)
            const int last_denied = 3; // Public per-op API uses the graph for every B.
#else
            const int last_denied = rows<=8?2:3;
#endif
            for(int denied=2;denied<=last_denied;++denied) {
                ledger.deny_call=ledger.calls+denied;
                require(run(out)==0 && ledger.live==0,"scratch/work reservation failure leaked staging or bypassed quota");
                ledger.deny_call=0;
            }
#if !defined(TSG_PUBLIC_API_ONLY)
            TSGgml_TestHostMoeFileReadFailNext();
            require(run(out)==0 && ledger.live==0,"worker read failure did not drain and refund staging");
            require(std::all_of(out.begin(),out.end(),[](float f){return f==-12345.f;}),"failed read changed output");
#endif
            // Original mapped addresses are deliberately wrong now. Exact output
            // proves the budgeted graph reads the registered files, not mmap.
            auto original_gate=gate; std::fill(gate.begin(),gate.end(),0);
            require(run(out)==1,"budgeted CPU graph failed");
            require(std::memcmp(ref.data(),out.data(),ref.size()*sizeof(float))==0,"compact CPU output differs from original full expert graph");
            require(ledger.live>0 && ledger.valid,"retained staging missing kind3 owner");
            const auto live=ledger.live;
            const auto calls=ledger.calls;
            require(run(out)==1 && ledger.live==live && ledger.calls==calls+1,"second layer did not reuse weight and work arenas");
            require(std::memcmp(ref.data(),out.data(),ref.size()*sizeof(float))==0,"reused graph output differs");
            std::copy(original_gate.begin(),original_gate.end(),gate.begin());
            std::int64_t released=0;
            require(TSGgml_TrimHostMoeExpertCacheScope(scope,0,&released)==1 && ledger.live==0,"trim retained host arena");
            ids[rows*K-1]=E;
            require(run(out)==0 && ledger.live==0,"invalid last route was not rejected"); ids.back()=2;
            routes.back()=std::numeric_limits<float>::infinity();
            require(run(out)==0 && ledger.live==0,"nonfinite last route was not rejected"); routes.back()=0.5f;
            require(TSGgml_EndHostMoeExpertCacheScope(scope,&released)==1 && ledger.live==0,"scope cleanup leaked"); scope=0;
            require(TSGgml_DetachSharedCacheBudget(&ledger)==1,"budget detach failed"); attached=false;
        }
#if defined(TSG_PUBLIC_API_ONLY)
        if(cuda) std::printf("PASS: %d checks, PUBLIC API compact CUDA configured shapes, ordinary-CUDA oracle, bounded CPU quota/duplicate fallback, reuse/trim; host peak %lld device peak %lld bytes\n",checks,static_cast<long long>(ledger.host_peak),static_cast<long long>(ledger.device_peak));
        else std::printf("PASS: %d checks, PUBLIC API B1/8/31/129/257/512/513 CPU graph bitexact, quota/commit/retry/trim; host peak %lld bytes\n",checks,static_cast<long long>(ledger.peak));
#else
        if(cuda) std::printf("PASS: %d checks, compact CUDA configured batch shapes, ordinary-CUDA oracle, bounded CPU quota/duplicate fallback, fault/drain/reuse/trim; host peak %lld device peak %lld bytes\n",checks,static_cast<long long>(ledger.host_peak),static_cast<long long>(ledger.device_peak));
        else std::printf("PASS: %d checks, N1/8 dedicated decode and N129/257/512/513 CPU full-graph bitexact, host peak %lld bytes\n",checks,static_cast<long long>(ledger.peak));
#endif
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s; native=%s\n",e.what(),TSGgml_GetLastError()); result=1; }
    if(scope) { std::int64_t n=0; if(!TSGgml_EndHostMoeExpertCacheScope(scope,&n)) result=1; }
    if(pipeline_scope && !TSGgml_EndHostMoeFilePipelineScope(pipeline_scope)) result=1;
    TSGgml_Shutdown();
    if(attached && !TSGgml_DetachSharedCacheBudget(&ledger)) result=1;
    std::error_code ec; std::filesystem::remove_all(dir,ec);
    return result;
}
