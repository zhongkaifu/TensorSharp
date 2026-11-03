// Copyright (c) Zhongkai Fu. All rights reserved.
// Synthetic standalone W8 contract. Production/BBD/model quality are not tested.
#include <cuda_runtime.h>
#include <cuda.h>
#include "reference.h"
#include "kernels.cuh"
#include "official.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
namespace fs=std::filesystem;
static void Rt(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>static bool Bits(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;}
static void Need(bool b,const char* e){if(!b)throw std::runtime_error(e);}
static std::string Q(const std::string& x){std::string y="\"";for(char c:x){if(c=='\\'||c=='\"')y+='\\';Need(c>=32,"control character in JSON");y+=c;}return y+'\"';}
static void Emit(const std::string& x){std::cout<<x<<std::endl;}
static const int Ts[]={1,7,8,14,16,28,32,33};
struct Shape{int rows,k,t,pattern;bool exact,real;};
static std::string Name(const Shape&s){return "r"+std::to_string(s.rows)+"-k"+std::to_string(s.k)+"-t"+std::to_string(s.t)+"-p"+std::to_string(s.pattern);}
static std::vector<Shape> Plan(){std::vector<Shape> a;for(int t:Ts){a.push_back({17,32,t,0,true,false});a.push_back({33,160,t,1,false,false});a.push_back({15,512,t,2,false,false});}for(int t:Ts)a.push_back({248320,5120,t,3,false,true});return a;}
static uint32_t Hash(uint32_t v){v^=v>>16;v*=0x7feb352d;v^=v>>15;v*=0x846ca68b;return v^(v>>16);}
struct Buffer{
    uint8_t*base=nullptr;size_t bytes;static constexpr size_t Guard=256;
    explicit Buffer(size_t n):bytes(n){Need(n&&n<SIZE_MAX-2*Guard,"allocation overflow");Rt(cudaMalloc(&base,n+2*Guard));Rt(cudaMemset(base,0xd3,n+2*Guard));}
    ~Buffer(){if(base)cudaFree(base);}Buffer(const Buffer&)=delete;
    void*Data()const{return base+Guard;}
    template<class T>T*At()const{return static_cast<T*>(Data());}
    void Check()const{std::array<uint8_t,Guard*2> got{};Rt(cudaMemcpy(got.data(),base,Guard,cudaMemcpyDeviceToHost));Rt(cudaMemcpy(got.data()+Guard,base+Guard+bytes,Guard,cudaMemcpyDeviceToHost));for(auto x:got)Need(x==0xd3,"device buffer guard changed");}
    template<class T>std::vector<T> Read()const{Need(bytes%sizeof(T)==0,"typed size");std::vector<T> a(bytes/sizeof(T));Rt(cudaMemcpy(a.data(),Data(),bytes,cudaMemcpyDeviceToHost));Check();return a;}
};
struct Events{cudaEvent_t a,b;Events(){Rt(cudaEventCreate(&a));Rt(cudaEventCreate(&b));}~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}};
static const void* Owned(int t){switch(t){
#define P(N) case N:return reinterpret_cast<const void*>(ts_w8_tc_##N);
 P(1) P(7) P(8) P(14) P(16) P(28) P(32) P(33)
#undef P
 default:throw std::runtime_error("unsupported token count");}}
static void Resource(const char*name,const void* f,int threads){cudaFuncAttributes a{};int active=0;Rt(cudaFuncGetAttributes(&a,f));Rt(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,threads,0));Need(active>=1&&a.maxThreadsPerBlock>=threads&&a.numRegs>0,"not launchable");Emit("{\"event\":\"resources\",\"symbol\":"+Q(name)+",\"registers\":"+std::to_string(a.numRegs)+",\"local_bytes\":"+std::to_string(a.localSizeBytes)+",\"shared_bytes\":"+std::to_string(a.sharedSizeBytes)+",\"active_ctas\":"+std::to_string(active)+",\"block_threads\":"+std::to_string(threads)+"}");}
static void Resources(){Resource("cast",(const void*)ts_w8_cast,256);Resource("reference_fp64",(const void*)ts_w8_reference,128);Resource("simt",(const void*)ts_w8_simt,128);Resource("sweep",(const void*)ts_w8_sweep,256);for(int t:Ts){std::string n="owned_"+std::to_string(t);Resource(n.c_str(),Owned(t),128);cudaFuncAttributes a{};int active=0,threads=0;Rt(OfficialAttributes(t,&a,&active,&threads));Need(active>=1&&threads>0&&a.maxThreadsPerBlock>=threads&&a.numRegs>0,"official not launchable");Emit("{\"event\":\"resources\",\"symbol\":\"official_"+std::to_string(t)+"\",\"registers\":"+std::to_string(a.numRegs)+",\"local_bytes\":"+std::to_string(a.localSizeBytes)+",\"shared_bytes\":"+std::to_string(a.sharedSizeBytes)+",\"active_ctas\":"+std::to_string(active)+",\"block_threads\":"+std::to_string(threads)+"}");}}
struct Sweep{
    std::unique_ptr<Buffer> data,checks;uint64_t words=0;int l2=0,blocks=0;uint64_t calls=0;
    Sweep(bool enabled,int sms){if(!enabled)return;Rt(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,0));Need(l2>0,"driver L2 unavailable");size_t bytes=(size_t(l2)*2+255)/256*256;words=bytes/4;blocks=sms*4;Need(blocks>0,"sweep blocks");data=std::make_unique<Buffer>(bytes);checks=std::make_unique<Buffer>(size_t(blocks)*16);Rt(cudaMemset(data->Data(),0x5a,bytes));Emit("{\"event\":\"sweep_setup\",\"driver_l2_bytes\":"+std::to_string(l2)+",\"sweep_bytes\":"+std::to_string(bytes)+",\"physical_cold_guaranteed\":false,\"method\":\"volatile_asm_ld_global_cg_u32_checksum_count\",\"all_data_evicted_control\":true}");}
    void Before(const Shape&s,const std::string&pair,int round,const char*half){if(!data)return;Rt(cudaMemset(checks->Data(),0xa5,checks->bytes));ts_w8_sweep<<<blocks,256>>>(data->At<uint32_t>(),words,checks->At<uint64_t>());Rt(cudaGetLastError());Rt(cudaDeviceSynchronize());auto got=checks->Read<uint64_t>();uint64_t sum=0,count=0,stride=uint64_t(blocks)*256;for(int b=0;b<blocks;b++){uint64_t left=words%stride,start=uint64_t(b)*256,expected=(words/stride)*256+(left>start?std::min<uint64_t>(256,left-start):0);Need(got[b*2]==expected*0x5a5a5a5aull&&got[b*2+1]==expected,"sweep per-block checksum/count");sum+=got[2*b];count+=got[2*b+1];}Need(sum==words*0x5a5a5a5aull&&count==words,"sweep coverage");data->Check();++calls;Emit("{\"event\":\"sweep\",\"shape\":"+Q(Name(s))+",\"pair\":"+Q(pair)+",\"round\":"+std::to_string(round)+",\"half\":"+Q(half)+",\"index\":"+std::to_string(calls)+",\"words_read\":"+std::to_string(count)+",\"checksum\":"+std::to_string(sum)+",\"before_event\":true,\"synchronized\":true,\"guards\":true}");}
};
static void Write(const fs::path&p,const void*ptr,size_t bytes){std::ofstream f(p,std::ios::binary|std::ios::out);Need(bool(f),"open output");f.write(static_cast<const char*>(ptr),bytes);Need(bool(f),"write output");}
template<class T>static void Write(const fs::path&p,const std::vector<T>&v){Write(p,v.data(),v.size()*sizeof(T));}
struct Case{
    Shape s;W8Layout l;std::vector<uint8_t>w;std::vector<float>x;std::vector<uint16_t>bx,cpuFp32;std::vector<double>ref,abs;
    Buffer dw,dx,db,dy,dr,da;std::vector<uint16_t>gold[3];
    explicit Case(Shape shape):s(shape),l(s.rows,s.k),w(l.total,0),x(size_t(s.t)*s.k),bx(x.size()),dw(l.total),dx(x.size()*4),db(x.size()*2),dy(size_t(s.t)*s.rows*2),dr(size_t(s.t)*s.rows*8),da(dr.bytes){
        static const uint16_t edge[]={0,0x8000,1,0x8001,0x0400,0x8400,0x3c00,0xbc00,0x7bff,0xfbff,0x3555};
        for(int r=0;r<s.rows;r++){for(uint64_t c=0;c<l.kpad;c++)w[l.Code(r,c)]=uint8_t(s.exact?int(Hash(uint32_t(r*131+c))%7)-3:Hash(uint32_t(r*131+c)));for(uint64_t g=0;g<l.groups;g++){uint16_t scale=s.exact?0x2000:s.pattern==2?edge[(r+g)%11]:uint16_t(0x3000+(Hash(uint32_t(r*173+g))%0x0800));WriteHalf(w,l.Scale(r,g),scale);(void)HalfValue(scale);}}
        for(size_t i=0;i<x.size();i++){x[i]=s.exact?float(int(i%7)-3):FValue(0x3d000000+(Hash(uint32_t(i))%0x01000000));if(!s.exact&&(i&1))x[i]=-x[i];if(s.pattern==2&&i%31==0)x[i]=FValue((i&1)?0x80000000:0);bx[i]=Bf16(x[i]);}
        Rt(cudaMemcpy(dw.Data(),w.data(),w.size(),cudaMemcpyHostToDevice));Rt(cudaMemcpy(dx.Data(),x.data(),x.size()*4,cudaMemcpyHostToDevice));Cast();Rt(cudaDeviceSynchronize());Need(db.Read<uint16_t>()==bx,"actual GPU BF16 input cast differs from independent CPU");
    }
    void Cast(){ts_w8_cast<<<(x.size()+255)/256,256>>>(dx.At<float>(),db.At<__nv_bfloat16>(),x.size());Rt(cudaGetLastError());}
    void Issue(int arm){Cast();if(arm==0)ts_w8_simt<<<(size_t(s.t)*s.rows+127)/128,128>>>(dw.At<int8_t>(),reinterpret_cast<__half*>(dw.At<uint8_t>()+l.scales_at),db.At<__nv_bfloat16>(),dy.At<__nv_bfloat16>(),s.rows,s.k,int(l.kpad),s.t);else if(arm==1){auto pw=dw.At<int8_t>();auto ps=reinterpret_cast<__half*>(dw.At<uint8_t>()+l.scales_at);auto px=db.At<__nv_bfloat16>();auto py=dy.At<__nv_bfloat16>();int r=s.rows,k=s.k,kp=int(l.kpad);void*args[]={&pw,&ps,&px,&py,&r,&k,&kp};Rt(cudaLaunchKernel(Owned(s.t),dim3((s.rows+15)/16),dim3(128),args));}else{Need(s.real,"official only original real geometry");Rt(LaunchOfficial(s.t,db.At<__nv_bfloat16>(),dw.At<int8_t>(),reinterpret_cast<__half*>(dw.At<uint8_t>()+l.scales_at),dy.At<__nv_bfloat16>()));}Rt(cudaGetLastError());}
    void Guards(){dw.Check();dx.Check();db.Check();dy.Check();dr.Check();da.Check();Need(db.Read<uint16_t>()==bx,"cast buffer content changed");}
    void Check(const std::vector<uint16_t>&out){CheckOutput(out,ref,abs,s.k,s.exact);}
    void Strict(const fs::path&data,bool dump){
        Rt(cudaMemset(dr.Data(),0xa5,dr.bytes));Rt(cudaMemset(da.Data(),0xa5,da.bytes));ts_w8_reference<<<(size_t(s.t)*s.rows+127)/128,128>>>(dw.At<int8_t>(),reinterpret_cast<__half*>(dw.At<uint8_t>()+l.scales_at),db.At<__nv_bfloat16>(),dr.At<double>(),da.At<double>(),s.rows,s.k,int(l.kpad),s.t);Rt(cudaGetLastError());Rt(cudaDeviceSynchronize());ref=dr.Read<double>();abs=da.Read<double>();for(size_t i=0;i<ref.size();i++)Need(std::isfinite(ref[i])&&std::isfinite(abs[i])&&abs[i]>=0,"invalid FP64 reference");
        if(!s.real){std::vector<double>cr,ca;std::vector<uint16_t>cf;CpuReference(w,l,bx,s.t,cr,ca,cf);Need(Bits(cr,ref)&&Bits(ca,abs),"independent CPU/GPU FP64 reference differs");cpuFp32=std::move(cf);}
        auto r0=ref,a0=abs;Rt(cudaMemset(dr.Data(),0x5a,dr.bytes));Rt(cudaMemset(da.Data(),0x5a,da.bytes));ts_w8_reference<<<(size_t(s.t)*s.rows+127)/128,128>>>(dw.At<int8_t>(),reinterpret_cast<__half*>(dw.At<uint8_t>()+l.scales_at),db.At<__nv_bfloat16>(),dr.At<double>(),da.At<double>(),s.rows,s.k,int(l.kpad),s.t);Rt(cudaGetLastError());Rt(cudaDeviceSynchronize());Need(Bits(dr.Read<double>(),r0)&&Bits(da.Read<double>(),a0),"reference dual-poison AA");
        for(int arm=0;arm<(s.real?3:2);arm++){for(int pass=0;pass<3;pass++){Rt(cudaMemset(dy.Data(),pass==1?0x5a:0xa5,dy.bytes));Issue(arm);Rt(cudaDeviceSynchronize());auto out=dy.Read<uint16_t>();Check(out);if(!s.real&&arm==0)Need(out==cpuFp32,"independent CPU FP32 group/FMA result differs from SIMT");if(!pass)gold[arm]=out;else Need(gold[arm]==out,"AA/dual poison full output bits");Guards();}}
        Need(dw.Read<uint8_t>()==w,"weight bytes mutated");Need(Bits(dx.Read<float>(),x),"input bytes mutated");
        if(dump){auto dir=data/Name(s);Need(fs::create_directory(dir),"fresh case data directory");Write(dir/"reference.f64",ref);Write(dir/"sumAbs.f64",abs);Write(dir/"input.f32",x);Write(dir/"input.bf16",bx);Write(dir/"simt.bf16",gold[0]);Write(dir/"owned.bf16",gold[1]);if(s.real)Write(dir/"official.bf16",gold[2]);}
        Emit("{\"event\":\"strict\",\"shape\":"+Q(Name(s))+",\"elements\":"+std::to_string(ref.size())+",\"independent_cpu_fp64\":"+(s.real?"false":"true")+",\"cpu_fp32_bits\":"+(s.real?"false":"true")+",\"numeric_bound\":true,\"exact_integer_case\":"+(s.exact?"true":"false")+",\"fullbits_AA\":true,\"dual_poison\":true,\"guards\":true,\"official\":"+(s.real?"true":"false")+",\"dumped\":"+(dump?"true":"false")+"}");
    }
    double Time(int arm,int iterations,Sweep&sweep,const std::string&pair,int round,const char*half){Rt(cudaMemset(dy.Data(),0xa5,dy.bytes));sweep.Before(s,pair,round,half);Events e;Rt(cudaEventRecord(e.a));for(int it=0;it<iterations;it++)Issue(arm);Rt(cudaEventRecord(e.b));Rt(cudaEventSynchronize(e.b));float ms=0;Rt(cudaEventElapsedTime(&ms,e.a,e.b));Need(std::isfinite(ms)&&ms>0,"nonpositive/nonfinite event time");auto out=dy.Read<uint16_t>();Need(out==gold[arm],"timed half changed full output bits");Check(out);Guards();return double(ms)/iterations;}
};
int main(int argc,char**argv){
    fs::path output,data;bool bench=false,cold=false;int rounds=10,iterations=1;
    try{for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--out"&&i+1<argc)output=argv[++i];else if(a=="--data"&&i+1<argc)data=argv[++i];else if(a=="--bench")bench=true;else if(a=="--cache-sweep")cold=true;else if(a=="--rounds"&&i+1<argc)rounds=std::stoi(argv[++i]);else if(a=="--iterations"&&i+1<argc)iterations=std::stoi(argv[++i]);else throw std::runtime_error("unknown/incomplete argument");}
        Need(!output.empty()&&!data.empty()&&rounds>=10&&iterations>0&&(!cold||(bench&&iterations==1)),"explicit outputs, rounds>=10; sweep iterations1");Need(fs::is_directory(output)&&!fs::exists(data)&&data.is_absolute()&&data.lexically_normal()==data&&fs::weakly_canonical(data)==data&&data.string().rfind("/dev/shm/",0)==0,"fresh physical SHM data and existing private output");fs::create_directories(data);
        int device=0,driver=0,runtime=0;Rt(cudaGetDevice(&device));Need(device==0,"device0 required");cudaDeviceProp p{};Rt(cudaGetDeviceProperties(&p,device));Rt(cudaDriverGetVersion(&driver));Rt(cudaRuntimeGetVersion(&runtime));char uuid[33];for(int i=0;i<16;i++)std::snprintf(uuid+2*i,3,"%02x",uint8_t(p.uuid.bytes[i]));Emit("{\"event\":\"device\",\"name\":"+Q(p.name)+",\"uuid\":"+Q(uuid)+",\"sms\":"+std::to_string(p.multiProcessorCount)+",\"driver\":"+std::to_string(driver)+",\"runtime\":"+std::to_string(runtime)+"}");Resources();Sweep sweep(cold,p.multiProcessorCount);
        auto plan=Plan();for(auto s:plan){Case c(s);c.Strict(data,true);}int timed=0;
        if(bench)for(int t:Ts){Shape s{248320,5120,t,3,false,true};Case c(s);c.Strict(data,false);for(auto pair:std::array<std::pair<int,const char*>,2>{{{0,"owned_vs_simt_functional"},{2,"owned_vs_official_primary"}}}){for(int round=0;round<rounds;round++){double a1=c.Time(pair.first,iterations,sweep,pair.second,round,"A1"),b1=c.Time(1,iterations,sweep,pair.second,round,"B1"),b2=c.Time(1,iterations,sweep,pair.second,round,"B2"),a2=c.Time(pair.first,iterations,sweep,pair.second,round,"A2");std::ostringstream o;o<<std::setprecision(12)<<"{\"event\":\"ABBA\",\"shape\":"<<Q(Name(s))<<",\"pair\":"<<Q(pair.second)<<",\"round\":"<<round<<",\"A1_ms\":"<<a1<<",\"B1_ms\":"<<b1<<",\"B2_ms\":"<<b2<<",\"A2_ms\":"<<a2<<",\"fullbits_each_half\":true,\"scope\":\"F32_to_BF16_cast_plus_W8_matmul_plus_BF16_store\"}";Emit(o.str());}}++timed;}
        std::ofstream f(output/"result.json");f<<"{\"complete\":true,\"numeric_pass\":true,\"strict_cases\":"<<plan.size()<<",\"timed_shapes\":"<<timed<<",\"rounds\":"<<rounds<<",\"iterations\":"<<iterations<<",\"cache_sweep\":"<<(cold?"true":"false")<<",\"sweeps\":"<<sweep.calls<<",\"model_quality\":\"NOT_RUN\"}\n";Need(bool(f),"result write");return 0;
    }catch(const std::exception&e){Emit("{\"event\":\"failure\",\"error\":"+Q(e.what())+"}");if(!output.empty()){std::ofstream f(output/"result.json");f<<"{\"complete\":false,\"numeric_pass\":false,\"model_quality\":\"NOT_RUN\"}\n";}return 1;}
}
