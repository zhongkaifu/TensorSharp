// Original recompiled Q6_K versus exact load-time int8 unpack. Not an actual production BBD oracle.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include "checks.h"
namespace fs=std::filesystem;
static fs::path Output;
static const char* TimingPlane="q6-lm-head";
#if defined(TS_Q6_BYTE_PLANES) && defined(TS_Q6_ALIGNED_BLOCKS)
#error mutually exclusive Q6 layouts
#endif
#ifdef TS_Q6_BYTE_PLANES
static constexpr bool BytePlanes=true;
#else
static constexpr bool BytePlanes=false;
#endif
#ifdef TS_Q6_ALIGNED_BLOCKS
static constexpr bool AlignedBlocks=true;
#else
static constexpr bool AlignedBlocks=false;
#endif
static const char* Layout=BytePlanes?"byte_planes_210":AlignedBlocks?"aligned_blocks_212":"int8_planes_274";
static std::vector<int> StrictNs(){std::vector<int> n{1,2,4,8,16,32,64};if(BytePlanes||AlignedBlocks)n.insert(n.end(),{7,14,28,9,17,33});return n;}
static std::vector<int> TimedNs(){std::vector<int> n{1,2,4,8,16,32,64};if(BytePlanes||AlignedBlocks)n.insert(n.end(),{7,14,28});return n;}
constexpr size_t Guard=64,GuardBytes=Guard*4;
struct Shape{int k,m,n,pattern;};
static void Rt(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Cu(CUresult e){if(e!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
static std::string Quote(const std::string& s){std::string t="\"";for(char c:s){if(c=='"'||c=='\\')t+='\\';if(c=='\n'){t+="\\n";continue;}t+=c;}return t+'"';}
static std::string Name(const Shape& s){return "k"+std::to_string(s.k)+"-m"+std::to_string(s.m)+"-n"+std::to_string(s.n)+"-p"+std::to_string(s.pattern);}
static uint32_t Hash(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;return uint32_t(x);}
static float Float(uint32_t b){float f;std::memcpy(&f,&b,4);return f;}
template<class T>static void Save(const fs::path& p,const std::vector<T>& a){std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));if(!f)throw std::runtime_error("evidence write failed");}
struct Buffer{
 void* base=nullptr;size_t bytes;
 explicit Buffer(size_t n):bytes(n){Rt(cudaMalloc(&base,n+2*GuardBytes));Rt(cudaMemset(base,0x56,n+2*GuardBytes));}
 ~Buffer(){if(base)cudaFree(base);}Buffer(const Buffer&)=delete;
 void* Data()const{return (uint8_t*)base+GuardBytes;}
 void Upload(const void* p){Rt(cudaMemcpy(Data(),p,bytes,cudaMemcpyHostToDevice));}
 template<class T>std::vector<T> Read()const{if(bytes%sizeof(T))throw std::runtime_error("read alignment");std::vector<T>a(bytes/sizeof(T));Rt(cudaMemcpy(a.data(),Data(),bytes,cudaMemcpyDeviceToHost));return a;}
 void CheckGuard()const{std::array<uint32_t,Guard>a{},b{};Rt(cudaMemcpy(a.data(),base,GuardBytes,cudaMemcpyDeviceToHost));Rt(cudaMemcpy(b.data(),(uint8_t*)Data()+bytes,GuardBytes,cudaMemcpyDeviceToHost));for(size_t i=0;i<Guard;i++)if(a[i]!=0x56565656u||b[i]!=0x56565656u)throw std::runtime_error("guard corruption");}
};
struct Variant{const char* symbol;CUfunction function;bool packed;};
struct Module{
 CUmodule module=nullptr;CUfunction quant=nullptr,pack=nullptr;std::array<CUfunction,4> oldMma{},newMma{};CUfunction oldMv=nullptr,newMv=nullptr;
 void Inspect(const char* name,CUfunction& f,int block){Cu(cuModuleGetFunction(&f,module,name));int regs,local,shared,maxThreads,active;Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f));Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f));Cu(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f));Cu(cuFuncGetAttribute(&maxThreads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f));Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,block,0));std::printf("{\"event\":\"driver_resources\",\"symbol\":%s,\"block_threads\":%d,\"registers\":%d,\"local_bytes\":%d,\"static_shared_bytes\":%d,\"active_ctas\":%d,\"max_block_threads\":%d}\n",Quote(name).c_str(),block,regs,local,shared,active,maxThreads);if(regs<=0||regs>255||local||maxThreads<block||active<1)throw std::runtime_error("actual driver resource gate rejected");}
 explicit Module(const fs::path& b){Cu(cuModuleLoad(&module,(b/"kernels.cubin").c_str()));Inspect("ts_q35_quantize_q81_f32",quant,256);Inspect("ts_bench_q6_prepack",pack,256);Inspect("ts_q35_gemv_q6k_1",oldMv,128);Inspect("ts_bench_gemv_q6packed_1",newMv,128);for(int i=0;i<4;i++){Inspect(("ts_q35_gemm_q6k_"+std::to_string(1<<i)).c_str(),oldMma[i],128);Inspect(("ts_bench_gemm_q6packed_"+std::to_string(1<<i)).c_str(),newMma[i],128);}}
 Variant Get(bool p,int n)const{if(n==1)return{p?"packed-gemv1":"oracle-gemv1",p?newMv:oldMv,p};int t=n<=8?0:n<=16?1:n<=32?2:3;return{p?"packed-mma":"oracle-mma",p?newMma[t]:oldMma[t],p};}
 ~Module(){if(module)cuModuleUnload(module);}
};
#include "cache-sweep.h"
struct Fixture{
 Shape shape;Module& module;Buffer raw,packed,input,q,d,out;std::vector<uint8_t> rawHost,expectedPacked,expectedQ;std::vector<uint32_t> expectedD;
 explicit Fixture(Module& mod,Shape s):shape(s),module(mod),raw(Q6Check::RawBytes(s.m,s.k)),packed(Q6Check::PackedBytes(s.m,s.k,BytePlanes,AlignedBlocks)),input(size_t(s.n)*s.k*4),q(size_t(s.n)*s.k),d(size_t(s.n)*(s.k/32)*4),out(size_t(s.n)*s.m*4){
  rawHost.resize(raw.bytes);std::vector<uint32_t>x(input.bytes/4);uint64_t seed=uint64_t(s.k)*131+s.m*17+s.pattern;
  static const uint16_t ds[]={0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0x0400,0x7c00,0xfc00,0x7e01,0xfe23};
  for(size_t b=0;b<raw.bytes/210;b++){
   uint8_t* p=rawHost.data()+b*210;for(int i=0;i<192;i++)p[i]=s.pattern==1?uint8_t(i*17):uint8_t(Hash(seed+b*210+i));
   for(int j=0;j<16;j++)p[192+j]=s.pattern==2?uint8_t(j%2?127:-128):s.pattern==3?uint8_t(j%3?0:-1):uint8_t(Hash(seed+b*29+j));
   uint16_t half=s.pattern==0?uint16_t(0x2000+(b%0x1000)):ds[(b+s.pattern)%std::size(ds)];std::memcpy(p+208,&half,2);
  }
  for(size_t i=0;i<x.size();i++){uint32_t h=Hash(seed+i);x[i]=s.pattern==3?(i%3?0:0x80000000u):((119u+h%8)<<23)|(h&0x807fffffu);}
  raw.Upload(rawHost.data());input.Upload(x.data());expectedPacked=BytePlanes?Q6Check::BytePlanes(rawHost,s.m,s.k):AlignedBlocks?Q6Check::AlignedBlocks(rawHost,s.m,s.k):Q6Check::Pack(rawHost,s.m,s.k);
  cudaEvent_t a,b;Rt(cudaEventCreate(&a));Rt(cudaEventCreate(&b));float load=0;std::array<float,2> packMs{};
  for(int pass=0;pass<2;pass++){
   Rt(cudaMemset(packed.Data(),pass?0x5a:0xa5,packed.bytes));void* src=raw.Data();void* dst=packed.Data();int m=s.m,k=s.k;void* args[]={&src,&dst,&m,&k};size_t work=BytePlanes?raw.bytes:AlignedBlocks?packed.bytes:size_t(m)*k;Rt(cudaEventRecord(a));Cu(cuLaunchKernel(module.pack,(work+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));Rt(cudaEventRecord(b));Rt(cudaEventSynchronize(b));Rt(cudaEventElapsedTime(&load,a,b));
   packMs[pass]=load;if(!std::isfinite(load)||load<=0)throw std::runtime_error("invalid prepack event interval");auto got=packed.Read<uint8_t>();Q6Check::Verify(rawHost,got,s.m,s.k,BytePlanes,AlignedBlocks);packed.CheckGuard();raw.CheckGuard();
  }
  Rt(cudaEventDestroy(b));Rt(cudaEventDestroy(a));
  if(!std::isfinite(load)||load<=0)throw std::runtime_error("invalid load-time event interval");
  for(int pass=0;pass<2;pass++){Rt(cudaMemset(q.Data(),pass?0x5a:0xa5,q.bytes));Rt(cudaMemset(d.Data(),pass?0x5a:0xa5,d.bytes));Quant();Rt(cudaDeviceSynchronize());auto qc=q.Read<uint8_t>();auto dc=d.Read<uint32_t>();if(pass==0){expectedQ=qc;expectedD=dc;}else if(qc!=expectedQ||dc!=expectedD)throw std::runtime_error("original quantizer dual-poison AA mismatch");}
  std::printf("{\"event\":\"prepack_pass\",\"shape\":%s,\"raw_bytes\":%zu,\"packed_bytes\":%zu,\"extra_bytes\":%zu,\"load_once_gpu_ms\":%.9g,\"prepack_GPU_ms_two_poison_passes\":[%.9g,%.9g],\"load_time_scope\":\"prepack_kernel_only_second_pass_not_full_model_load\",\"whole_bytes_CPU_decoder\":true,\"dual_poison\":true,\"guards\":true}\n",Quote(Name(s)).c_str(),raw.bytes,packed.bytes,packed.bytes-raw.bytes,load,packMs[0],packMs[1]);
 }
 void Quant(){void* x=input.Data();void* xq=q.Data();void* xd=d.Data();int k=shape.k,n=shape.n;void* args[]={&x,&xq,&xd,&k,&n};Cu(cuLaunchKernel(module.quant,(size_t(n)*k+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));}
 void Reset(uint8_t poison){Rt(cudaMemset(out.Data(),poison,out.bytes));}
 void Issue(const Variant& v,int splits=1){Quant();void* w=v.packed?packed.Data():raw.Data();long long stride=(shape.k/256)*(v.packed&&AlignedBlocks?212:210);void* xq=q.Data();void* xd=d.Data();void* y=out.Data();int m=shape.m,k=shape.k,n=shape.n;if(splits>1)Rt(cudaMemsetAsync(y,0,out.bytes));if(n==1){void* args[]={&w,&stride,&m,&k,&xq,&xd,&y};Cu(cuLaunchKernel(v.function,(m+3)/4,1,1,128,1,1,0,nullptr,args,nullptr));}else{void* args[]={&w,&stride,&m,&k,&xq,&xd,&n,&y};Cu(cuLaunchKernel(v.function,(m+63)/64,splits,1,128,1,1,0,nullptr,args,nullptr));}}
 std::vector<uint32_t> Read(){Rt(cudaDeviceSynchronize());auto v=out.Read<uint32_t>();for(Buffer* b:{&raw,&packed,&input,&q,&d,&out})b->CheckGuard();if(q.Read<uint8_t>()!=expectedQ||d.Read<uint32_t>()!=expectedD)throw std::runtime_error("quantized bytes/scales changed");return v;}
 void Immutable(){if(raw.Read<uint8_t>()!=rawHost||packed.Read<uint8_t>()!=expectedPacked)throw std::runtime_error("immutable weight payload changed");}
};
static void Equal(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,const Shape& s,const char* phase){auto at=std::mismatch(a.begin(),a.end(),b.begin());if(at.first==a.end())return;Save(Output/"failure-reference.f32",a);Save(Output/"failure-actual.f32",b);std::ofstream f(Output/"failure.json");f<<"{\"shape\":"<<Quote(Name(s))<<",\"phase\":"<<Quote(phase)<<",\"index\":"<<(at.first-a.begin())<<",\"reference\":"<<*at.first<<",\"actual\":"<<*at.second<<"}\n";throw std::runtime_error("whole F32 bits differ: "+Name(s)+" "+phase);}
static std::vector<uint32_t> Strict(Fixture& f){auto a=f.module.Get(false,f.shape.n),b=f.module.Get(true,f.shape.n);f.Reset(0xa5);f.Issue(a);auto ref=f.Read();f.Reset(0x5a);f.Issue(a);Equal(ref,f.Read(),f.shape,"oracle dual-poison A/A");f.Reset(0xa5);f.Issue(b);auto got=f.Read();Equal(ref,got,f.shape,"candidate");f.Reset(0x5a);f.Issue(b);Equal(got,f.Read(),f.shape,"candidate dual-poison A/A");f.Immutable();return ref;}
static float Timed(Fixture& f,const Variant& v,const std::vector<uint32_t>& ref,int iterations,CacheSweep* sweep,int round,const char* half){f.Reset(0xa5);cudaEvent_t a,b;Rt(cudaEventCreate(&a));Rt(cudaEventCreate(&b));if(sweep)sweep->Before(f.shape,v,"packed",round,half);Rt(cudaEventRecord(a));for(int i=0;i<iterations;i++)f.Issue(v);Rt(cudaEventRecord(b));Rt(cudaEventSynchronize(b));float ms;Rt(cudaEventElapsedTime(&ms,a,b));Rt(cudaEventDestroy(b));Rt(cudaEventDestroy(a));Equal(ref,f.Read(),f.shape,"timed-half");if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid event timing");return ms/iterations;}
int main(int argc,char**argv)try{
 if(argc<4)throw std::runtime_error("BUILD OUT --strict|--bench [--cache-sweep] [--rounds 10] [--iterations 3] [--atomic-aa]");fs::path build=fs::canonical(argv[1]);Output=fs::canonical(argv[2]);std::string mode=argv[3];bool sweepOn=false,atomic=false;int rounds=10,iterations=3;bool explicitIterations=false;
 for(int i=4;i<argc;i++){std::string a=argv[i];if(a=="--cache-sweep")sweepOn=true;else if(a=="--atomic-aa")atomic=true;else if(a=="--rounds"&&i+1<argc)rounds=std::stoi(argv[++i]);else if(a=="--iterations"&&i+1<argc){iterations=std::stoi(argv[++i]);explicitIterations=true;}else throw std::runtime_error("unknown runner option");}
 if(sweepOn&&!explicitIterations)iterations=1;if((mode!="--strict"&&mode!="--bench")||rounds<10||iterations<1||(sweepOn&&(mode!="--bench"||iterations!=1)))throw std::runtime_error("invalid fixed gate/timing settings");
 Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));cudaDeviceProp dp{};Rt(cudaGetDeviceProperties(&dp,0));int driver,runtime;Rt(cudaDriverGetVersion(&driver));Rt(cudaRuntimeGetVersion(&runtime));std::printf("{\"event\":\"device\",\"name\":%s,\"sms\":%d,\"driver\":%d,\"runtime\":%d}\n",Quote(dp.name).c_str(),dp.multiProcessorCount,driver,runtime);Module module(build);
 size_t strict=0;for(int n:StrictNs())for(int p=0;p<4;p++)for(int kk:{256,512,5120}){Shape s{kk,p%2?65:63,n,p};Fixture f(module,s);Strict(f);++strict;std::printf("{\"event\":\"strict_pass\",\"shape\":%s,\"full_bits\":true,\"dual_poison_AA\":true,\"guards\":true,\"split_k\":1}\n",Quote(Name(s)).c_str());}
 std::unique_ptr<CacheSweep>sweep;if(sweepOn)sweep=std::make_unique<CacheSweep>(module,dp.multiProcessorCount);size_t timed=0;
 for(int n:TimedNs()){Shape s{5120,248320,n,0};Fixture f(module,s);auto ref=Strict(f);++strict;std::printf("{\"event\":\"strict_pass\",\"shape\":%s,\"full_bits\":true,\"dual_poison_AA\":true,\"guards\":true,\"split_k\":1}\n",Quote(Name(s)).c_str());
  if(atomic&&n>1){for(bool p:{false,true}){auto v=module.Get(p,n);f.Reset(0xa5);f.Issue(v,2);auto a=f.Read();f.Reset(0x5a);f.Issue(v,2);auto b=f.Read();size_t diff=0;for(size_t i=0;i<a.size();i++)diff+=a[i]!=b[i];std::printf("{\"event\":\"atomic_AA_DIAGNOSTIC_NOT_EQUIVALENCE_PASS\",\"shape\":%s,\"packed\":%s,\"split_k\":2,\"differing_F32_bits\":%zu}\n",Quote(Name(s)).c_str(),p?"true":"false",diff);}}
 }
 if(mode=="--bench")for(int n:TimedNs()){Shape s{5120,248320,n,0};Fixture f(module,s);auto ref=Strict(f);std::printf("{\"event\":\"pipeline_pass\",\"shape\":%s,\"full_bits_AA_guards\":true}\n",Quote(Name(s)).c_str());{auto a=module.Get(false,n),b=module.Get(true,n);for(int r=0;r<rounds;r++){float a1=Timed(f,a,ref,iterations,sweep.get(),r,"A1"),b1=Timed(f,b,ref,iterations,sweep.get(),r,"B1"),b2=Timed(f,b,ref,iterations,sweep.get(),r,"B2"),a2=Timed(f,a,ref,iterations,sweep.get(),r,"A2");std::printf("{\"event\":\"ABBA_gpu_ms\",\"shape\":%s,\"n\":%d,\"round\":%d,\"A1_oracle_ms\":%.9g,\"B1_candidate_ms\":%.9g,\"B2_candidate_ms\":%.9g,\"A2_oracle_ms\":%.9g,\"fullbits_each_half\":true,\"scope\":\"original_Q81_quant_plus_zero_if_needed_plus_Q6_matmul\",\"split_k\":1}\n",Quote(Name(s)).c_str(),n,r,a1,b1,b2,a2);}timed++;}f.Immutable();}
 std::ofstream result(Output/"result.json");result<<"{\"complete\":true,\"strict_pass\":true,\"layout\":"<<Quote(Layout)<<",\"strict_cases\":"<<strict<<",\"timed_shapes\":"<<timed<<",\"rounds\":"<<rounds<<",\"iterations\":"<<iterations<<",\"cache_sweep\":"<<(sweepOn?"true":"false")<<",\"cache_sweeps\":"<<(sweep?sweep->calls:0)<<",\"production_atomic_equivalence\":\"NOT_ESTABLISHED\",\"actual_BBD_or_model_gate\":\"NOT_RUN\"}\n";return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());if(!Output.empty()){std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"strict_pass\":false,\"error\":"<<Quote(e.what())<<"}\n";}return 1;}
