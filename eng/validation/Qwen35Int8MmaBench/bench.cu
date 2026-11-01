// Standalone driver runner. Every --bench run executes the complete strict gate
// before any GPU-event ABBA samples. No production dispatch or relaxed tolerance.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
struct Shape { int k, m, pattern; uint32_t alpha; bool timed; };
#include "plan.h"
static fs::path Output;
constexpr size_t Guard = 64, GuardBytes = Guard * 4;
constexpr uint32_t Canary = 0x4f123456u;
static void Rt(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void Cu(CUresult e) { if(e!=CUDA_SUCCESS) { const char* s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"CUDA driver error"); } }
static float Float(uint32_t bits) { float f;std::memcpy(&f,&bits,4);return f; }
static uint32_t Hash(uint64_t x) { x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;return uint32_t(x); }
static std::string Quote(const std::string& s) { std::string o="\"";for(char c:s) {if(c=='\\'||c=='\"')o+='\\';if(c=='\n'){o+="\\n";continue;}if((unsigned char)c<32){o+='?';continue;}o+=c;}return o+'\"'; }
static std::string Name(const Shape& s) { return "k"+std::to_string(s.k)+"-m"+std::to_string(s.m)+"-p"+std::to_string(s.pattern)+"-alpha"+std::to_string(s.alpha); }
static void Save(const fs::path& p,const std::vector<uint32_t>& values) {std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*4);if(!f)throw std::runtime_error("failed writing bit evidence");}

struct Buffer {
    void* base=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n) {Rt(cudaMalloc(&base,n+2*GuardBytes));Rt(cudaMemset(base,0x56,n+2*GuardBytes));}
    Buffer(const Buffer&)=delete;
    ~Buffer(){if(base)cudaFree(base);}
    void* Data()const{return static_cast<uint8_t*>(base)+GuardBytes;}
    void Upload(const void* host){Rt(cudaMemcpy(Data(),host,bytes,cudaMemcpyHostToDevice));}
    void CheckGuard()const {
        std::array<uint32_t,Guard> a{},b{};
        Rt(cudaMemcpy(a.data(),base,GuardBytes,cudaMemcpyDeviceToHost));
        Rt(cudaMemcpy(b.data(),static_cast<uint8_t*>(Data())+bytes,GuardBytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Guard;i++)if(a[i]!=0x56565656u||b[i]!=0x56565656u)throw std::runtime_error("input guard corruption");
    }
};
struct Module {
    CUmodule module=nullptr;CUfunction oracle=nullptr,candidate=nullptr;
    explicit Module(const fs::path& build) {
        Cu(cuModuleLoad(&module,(build/"kernels.cubin").c_str()));
        Cu(cuModuleGetFunction(&oracle,module,"ts_q35_gemv_nvfp4_8"));
        Cu(cuModuleGetFunction(&candidate,module,"ts_bench_gemv_nvfp4_int8mma_8"));
        int regs=0,local=0,shared=0,threads=0,active=0,binary=0;
        Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,candidate));
        Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,candidate));
        Cu(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,candidate));
        Cu(cuFuncGetAttribute(&threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,candidate));
        Cu(cuFuncGetAttribute(&binary,CU_FUNC_ATTRIBUTE_BINARY_VERSION,candidate));
        Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,candidate,1024,0));
        std::printf("{\"event\":\"driver_resources\",\"registers\":%d,\"local_bytes\":%d,\"static_shared_bytes\":%d,\"max_block_threads\":%d,\"active_ctas\":%d,\"binary_version\":%d}\n",regs,local,shared,threads,active,binary);
        if(regs<=0||regs>64||local!=0||shared<16508||shared>16512||threads<1024||active<1)
            throw std::runtime_error("actual driver resource gate rejected candidate; no quality comparison permitted");
    }
    ~Module(){if(module)cuModuleUnload(module);}
};

struct Fixture {
    Shape shape;size_t count;Buffer codes,scales,xq,xd,out;
    explicit Fixture(Shape s):shape(s),count(size_t(8)*s.m),codes(size_t(s.m)*(s.k/2)),
        scales(size_t(s.m)*(s.k/16)),xq(size_t(8)*s.k),xd(size_t(8)*(s.k/32)*4),out(count*4) {
        if(s.k<=0||s.k%64||s.m<=0)throw std::runtime_error("unsupported K/outDim shape");
        std::vector<uint8_t> c(codes.bytes),d(scales.bytes),q(xq.bytes);
        std::vector<float> x(xd.bytes/4);
        uint64_t seed=0x20261101ull+uint64_t(s.k)*37+uint64_t(s.m)*19+uint64_t(s.pattern)*101;
        for(size_t i=0;i<c.size();i++) {
            const int b=int((i%(s.k/2))/32),sub=int((i%32)/8);
            c[i]=s.pattern==1?((i/(s.k/2))&1?0xff:0x77):s.pattern==2?((b+sub)&1?0xff:0x77):
                 s.pattern==3?uint8_t(i*17u):s.pattern==4?0:s.pattern==5?0x88:uint8_t(Hash(seed+i));
        }
        if(s.pattern==4)for(int r=0;r<s.m;r++)for(int b=0;b<s.k/64;b++)for(int sub=0;sub<4;sub++) {
            int logical=(r+b+sub)%16;
            size_t at=size_t(r)*(s.k/2)+b*32+sub*8+(logical%8);
            c[at]|=uint8_t(((r+sub)&1?15:7)<<(logical>=8?4:0));
        }
        const uint8_t boundaries[]={0,1,7,8,0x77,0x7e,0x7f,0x80,0x81,0xfe,0xff};
        for(size_t i=0;i<d.size();i++)d[i]=(s.pattern==1||s.pattern==3||s.pattern==5)?boundaries[i%11]:uint8_t(Hash(seed+0x10000000ull+i));
        const int signedCases[]={-128,-127,-1,0,1,127};
        for(size_t i=0;i<q.size();i++)q[i]=s.pattern==1?uint8_t((i/s.k)&1?-128:127):
            (s.pattern==3||s.pattern==5)?uint8_t(signedCases[i%6]):uint8_t(Hash(seed+0x20000000ull+i));
        const uint32_t small[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
            0x00800000u,0x80800000u,0x3f800001u,0xbf800001u,0x67800000u,0xe7800000u};
        for(size_t i=0;i<x.size();i++) {
            uint32_t h=Hash(seed+0x30000000ull+i);
            uint32_t bits=(s.pattern==3||s.pattern==5)?small[i%12]:s.pattern==2?
                ((i%3==0)?0x3f800001u:(i%3==1)?0x3f800000u:0x33800000u):
                ((115u+h%11)<<23)|(h&0x7fffffu);
            x[i]=Float(bits);
        }
        codes.Upload(c.data());scales.Upload(d.data());xq.Upload(q.data());xd.Upload(x.data());
    }
    void Reset() {std::vector<uint32_t> v(count+2*Guard,Canary);Rt(cudaMemcpy(out.base,v.data(),v.size()*4,cudaMemcpyHostToDevice));}
    void Issue(CUfunction fn,bool candidate) {
        void* c=codes.Data();void* s=scales.Data();void* q=xq.Data();void* d=xd.Data();void* y=out.Data();
        int m=shape.m,k=shape.k;float alpha=Float(shape.alpha);
        void* args[]={&c,&s,&m,&k,&q,&d,&alpha,&y};
        Cu(cuLaunchKernel(fn,(m+15)/16,1,1,candidate?1024:128,1,1,0,nullptr,args,nullptr));
    }
    std::vector<uint32_t> Read() {
        Rt(cudaDeviceSynchronize());std::vector<uint32_t> v(count+2*Guard);
        Rt(cudaMemcpy(v.data(),out.base,v.size()*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<Guard;i++)if(v[i]!=Canary||v[v.size()-1-i]!=Canary) {
            Save(Output/"failure-guard-output.f32",v);throw std::runtime_error("output guard corruption");
        }
        for(size_t i=Guard;i<Guard+count;i++)if(v[i]==Canary) {
            Save(Output/"failure-unwritten-output.f32",v);throw std::runtime_error("unwritten output sentinel");
        }
        codes.CheckGuard();scales.CheckGuard();xq.CheckGuard();xd.CheckGuard();
        return v;
    }
};
static void Equal(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,const Shape& s,const char* phase) {
    auto diff=std::mismatch(a.begin(),a.end(),b.begin());if(diff.first==a.end())return;
    size_t at=size_t(diff.first-a.begin());Save(Output/"failure-reference-output.f32",a);Save(Output/"failure-actual-output.f32",b);
    std::ofstream f(Output/"failure.json");f<<"{\"shape\":"<<Quote(Name(s))<<",\"phase\":"<<Quote(phase)
        <<",\"index_including_guard\":"<<at<<",\"reference_bits\":"<<a[at]<<",\"actual_bits\":"<<b[at]<<"}\n";
    throw std::runtime_error("complete-output bit mismatch: "+Name(s)+" "+phase);
}
static void Strict(Fixture& f,Module& module) {
    f.Reset();f.Issue(module.oracle,false);auto a=f.Read();
    f.Reset();f.Issue(module.oracle,false);Equal(a,f.Read(),f.shape,"oracle_A-A");
    f.Reset();f.Issue(module.candidate,true);auto b=f.Read();Equal(a,b,f.shape,"oracle-candidate");
    f.Reset();f.Issue(module.candidate,true);Equal(b,f.Read(),f.shape,"candidate_A-A");
}
static float Time(Fixture& f,CUfunction fn,bool candidate,int iterations) {
    cudaEvent_t begin,end;Rt(cudaEventCreate(&begin));Rt(cudaEventCreate(&end));
    Rt(cudaEventRecord(begin));for(int i=0;i<iterations;i++)f.Issue(fn,candidate);
    Rt(cudaEventRecord(end));Rt(cudaEventSynchronize(end));float ms=0;Rt(cudaEventElapsedTime(&ms,begin,end));
    Rt(cudaEventDestroy(end));Rt(cudaEventDestroy(begin));
    if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid GPU-event timing");return ms/iterations;
}
int main(int argc,char** argv) try {
    if(argc<4)throw std::runtime_error("usage: int8mma-bench BUILD_DIR OUT_DIR --strict|--bench [--acts8] [--rounds5] [--iterations3]");
    fs::path build=fs::weakly_canonical(argv[1]);Output=fs::weakly_canonical(argv[2]);std::string mode=argv[3];
    auto relative=Output.lexically_relative(build);
    if(relative.empty()||*relative.begin()=="..")throw std::runtime_error("OUT_DIR must be below BUILD_DIR");
    fs::create_directories(Output);
    int acts=8,rounds=5,iterations=3;
    for(int i=4;i<argc;i+=2) {
        if(i+1>=argc)throw std::runtime_error("missing option value");
        std::string option=argv[i];int value=std::stoi(argv[i+1]);
        if(option=="--acts")acts=value;else if(option=="--rounds")rounds=value;else if(option=="--iterations")iterations=value;else throw std::runtime_error("unknown option");
    }
    if(acts!=8) {std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"status\":\"NOT_SUPPORTED\",\"activation_rows\":"<<acts<<",\"gpu_invoked\":false}\n";return 2;}
    if((mode!="--strict"&&mode!="--bench")||rounds<1||iterations<1)throw std::runtime_error("invalid mode/rounds/iterations");
    {std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"status\":\"GPU_GATE_IN_PROGRESS\"}\n";}
    Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));
    cudaDeviceProp device{};Rt(cudaGetDeviceProperties(&device,0));int driver=0,runtime=0;Rt(cudaDriverGetVersion(&driver));Rt(cudaRuntimeGetVersion(&runtime));
    std::printf("{\"event\":\"device\",\"name\":%s,\"cc_major\":%d,\"cc_minor\":%d,\"sms\":%d,\"registers_per_sm\":%d,\"driver\":%d,\"runtime\":%d}\n",Quote(device.name).c_str(),device.major,device.minor,device.multiProcessorCount,device.regsPerMultiprocessor,driver,runtime);
    Module module(build);
    {Fixture pilot({64,1,5,0x80000000u,false});Strict(pilot,module);}
    std::printf("{\"event\":\"driver_launch_gate\",\"launchable\":true,\"full_quality_plan_passed\":false}\n");
    size_t strict=0,timed=0;
    for(const Shape& s:StrictPlan) {
        Fixture f(s);Strict(f,module);strict++;
        std::printf("{\"event\":\"strict_case_pass\",\"shape\":%s,\"k\":%d,\"out_dim\":%d,\"pattern\":%d,\"alpha_bits\":%u,\"full_output_bits\":true,\"outputs\":%zu,\"oracle_AA\":true,\"candidate_AA\":true,\"guards\":true}\n",Quote(Name(s)).c_str(),s.k,s.m,s.pattern,s.alpha,f.count);
    }
    if(mode=="--bench")for(const Shape& s:StrictPlan)if(s.timed) {
        Fixture f(s);Strict(f,module); // exact timed fixture must pass again
        for(int i=0;i<2;i++){f.Issue(module.oracle,false);f.Issue(module.candidate,true);}Rt(cudaDeviceSynchronize());
        for(int round=0;round<rounds;round++) {
            float a=Time(f,module.oracle,false,iterations),b=Time(f,module.candidate,true,iterations);
            float c=Time(f,module.candidate,true,iterations),d=Time(f,module.oracle,false,iterations);
            std::printf("{\"event\":\"ABBA_gpu_ms\",\"shape\":%s,\"k\":%d,\"out_dim\":%d,\"round\":%d,\"iterations_per_sample\":%d,\"A1_oracle_ms\":%.9g,\"B1_candidate_ms\":%.9g,\"B2_candidate_ms\":%.9g,\"A2_oracle_ms\":%.9g}\n",Quote(Name(s)).c_str(),s.k,s.m,round,iterations,a,b,c,d);
        }
        f.Read();timed++;
    }
    std::ofstream f(Output/"result.json");f<<"{\"complete\":true,\"status\":\"STRICT_OUTPUT_BITS_PASS\",\"activation_rows\":8,\"strict_cases\":"<<strict<<",\"timed_shapes\":"<<timed
        <<",\"rounds\":"<<rounds<<",\"iterations_per_sample\":"<<iterations<<",\"hardware_counters\":\"NOT_RUN\",\"production_integrated\":false}\n";
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what());
    if(!Output.empty()) {std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"status\":\"REJECTED_NOT_VALIDATED\",\"error\":"<<Quote(error.what())<<"}\n";}
    return 1;
}
