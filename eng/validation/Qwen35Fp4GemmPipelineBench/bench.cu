// Standalone driver harness. Each invocation tests one private pipeline arm.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "fixtures.h"
#include "checks.h"
namespace fs=std::filesystem;
#include "plan.h"
struct Variant {int id,bn,stages,bk,smem;const char*name[2];CUfunction fn[2];};
#include "variants.h"
static fs::path Output;
constexpr size_t Guard=64,GuardBytes=Guard*4;
constexpr uint32_t GuardWord=0x56565656u,PoisonA=0x7fa12345u,PoisonB=0xff812345u;
static void Rt(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Cu(CUresult e){if(e!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"CUDA driver error");}}
static float Float(uint32_t bits){float f;std::memcpy(&f,&bits,4);return f;}
static std::string Quote(const std::string&s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';if(c=='\n'){o+="\\n";continue;}if((unsigned char)c<32){o+='?';continue;}o+=c;}return o+'\"';}
static void SaveBits(const fs::path&p,const std::vector<uint32_t>&x){std::ofstream f(p,std::ios::binary);f.write((const char*)x.data(),x.size()*4);if(!f)throw std::runtime_error("failure evidence write");}
struct Buffer {
 void*base=nullptr;size_t bytes;
 explicit Buffer(size_t n):bytes(n){Rt(cudaMalloc(&base,n+2*GuardBytes));Rt(cudaMemset(base,0x56,n+2*GuardBytes));}
 Buffer(const Buffer&)=delete;~Buffer(){if(base)cudaFree(base);}
 void*Data()const{return static_cast<uint8_t*>(base)+GuardBytes;}
 void Upload(const void*p){Rt(cudaMemcpy(Data(),p,bytes,cudaMemcpyHostToDevice));}
 void Check()const{std::array<uint32_t,Guard>a{},b{};Rt(cudaMemcpy(a.data(),base,GuardBytes,cudaMemcpyDeviceToHost));Rt(cudaMemcpy(b.data(),static_cast<uint8_t*>(Data())+bytes,GuardBytes,cudaMemcpyDeviceToHost));for(size_t i=0;i<Guard;i++)if(a[i]!=GuardWord||b[i]!=GuardWord)throw std::runtime_error("input guard corruption");}
 void Unchanged(const void*expected)const{Check();std::vector<uint8_t>actual(bytes);Rt(cudaMemcpy(actual.data(),Data(),bytes,cudaMemcpyDeviceToHost));if(std::memcmp(actual.data(),expected,bytes))throw std::runtime_error("input payload mutated");}
};
struct Module {
 CUmodule module=nullptr;
 explicit Module(const fs::path&build,int chosen){
  Cu(cuModuleLoad(&module,(build/"kernels.cubin").c_str()));
  CUfunction query=nullptr;Cu(cuModuleGetFunction(&query,module,"ts_bench_fp4_smem_query"));Buffer buf(5*sizeof(int));void*data=buf.Data();void*args[]={&data};Cu(cuLaunchKernel(query,1,1,1,1,1,1,0,nullptr,args,nullptr));std::array<int,5>smem{};Rt(cudaMemcpy(smem.data(),data,sizeof(smem),cudaMemcpyDeviceToHost));buf.Check();
  for(int i=0;i<5;i++)if(smem[i]!=Variants[i].smem)throw std::runtime_error("actual sizeof shared differs from descriptor");
  std::printf("{\"event\":\"shared_size_query\",\"guards\":true,\"actual_bytes\":[%d,%d,%d,%d,%d]}\n",smem[0],smem[1],smem[2],smem[3],smem[4]);
  for(int id:{0,chosen})for(int fused=0;fused<2;fused++)Inspect(Variants[id],fused);
 }
 void Inspect(Variant&v,int fused){
  CUfunction&f=v.fn[fused];Cu(cuModuleGetFunction(&f,module,v.name[fused]));int regs=0,local=0,shared=0,maxThreads=0,active=0,binary=0,maxDynamic=0;
  Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,f));Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f));Cu(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,f));Cu(cuFuncGetAttribute(&maxThreads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,f));Cu(cuFuncGetAttribute(&binary,CU_FUNC_ATTRIBUTE_BINARY_VERSION,f));
  Cu(cuFuncSetAttribute(f,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,v.smem));Cu(cuFuncGetAttribute(&maxDynamic,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,f));Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,256,v.smem));
  std::printf("{\"event\":\"driver_resources\",\"variant\":%d,\"symbol\":%s,\"swiglu\":%s,\"registers\":%d,\"local_bytes\":%d,\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d,\"total_shared_bytes\":%d,\"max_dynamic_shared_bytes\":%d,\"max_block_threads\":%d,\"block_threads\":256,\"active_ctas\":%d,\"binary_version\":%d}\n",v.id,Quote(v.name[fused]).c_str(),fused?"true":"false",regs,local,shared,v.smem,shared+v.smem,maxDynamic,maxThreads,active,binary);
  if(maxThreads<256||maxDynamic<v.smem||active<1)throw std::runtime_error("actual driver launchability rejected; no 2CTA assumption");
 }
 ~Module(){if(module)cuModuleUnload(module);}
};
struct Fixture {
 Shape shape;PipelineInputs host;size_t count;Buffer wc,ws,xc,xs,scale,out;
 explicit Fixture(Shape s):shape(s),host(s),count(size_t(s.rows)*(s.swiglu?s.out_dim/2:s.out_dim)),wc(host.wc.size()),ws(host.ws.size()),xc(host.xc.size()),xs(host.xs.size()),scale(host.scale.size()*4),out(count*4){wc.Upload(host.wc.data());ws.Upload(host.ws.data());xc.Upload(host.xc.data());xs.Upload(host.xs.data());scale.Upload(host.scale.data());}
 void Reset(uint32_t poison){std::vector<uint32_t>words(count+2*Guard,GuardWord);std::fill(words.begin()+Guard,words.begin()+Guard+count,poison);Rt(cudaMemcpy(out.base,words.data(),words.size()*4,cudaMemcpyHostToDevice));}
 void Issue(Variant&v){void*w=wc.Data(),*s=ws.Data(),*x=xc.Data(),*t=xs.Data(),*r=scale.Data(),*y=out.Data(),*partial=nullptr,*counters=nullptr;int o=shape.out_dim,k=shape.k,m=shape.rows;float alpha=Float(shape.alpha);void*args[]={&w,&s,&o,&k,&x,&t,&r,&m,&alpha,&y,&partial,&counters};Cu(cuLaunchKernel(v.fn[shape.swiglu?1:0],(m+v.bn-1)/v.bn,(o+127)/128,1,256,1,1,v.smem,nullptr,args,nullptr));}
 std::vector<uint32_t>Read(){Rt(cudaDeviceSynchronize());std::vector<uint32_t>x(count+2*Guard);Rt(cudaMemcpy(x.data(),out.base,x.size()*4,cudaMemcpyDeviceToHost));PipelineGuards(x,count,Guard,GuardWord);wc.Unchanged(host.wc.data());ws.Unchanged(host.ws.data());xc.Unchanged(host.xc.data());xs.Unchanged(host.xs.data());scale.Unchanged(host.scale.data());return x;}
};
static void Equal(const std::vector<uint32_t>&a,const std::vector<uint32_t>&b,const Shape&s,const char*phase){size_t i=PipelineDifference(a,b);if(i==a.size())return;SaveBits(Output/"failure-reference.f32",a);SaveBits(Output/"failure-actual.f32",b);std::ofstream f(Output/"failure.json");f<<"{\"shape\":"<<Quote(PipelineName(s))<<",\"phase\":"<<Quote(phase)<<",\"guarded_index\":"<<i<<",\"reference_bits\":"<<a[i]<<",\"actual_bits\":"<<b[i]<<"}\n";throw std::runtime_error("whole-output bits/coverage mismatch; no finite/tolerance bypass");}
static std::vector<uint32_t>Oracle(Fixture&f){f.Reset(PoisonA);f.Issue(Variants[0]);auto a=f.Read();f.Reset(PoisonB);f.Issue(Variants[0]);Equal(a,f.Read(),f.shape,"oracle_AA_dual_poison_complete_write");return a;}
static void Strict(Fixture&f,Variant&v,const std::vector<uint32_t>&oracle){f.Reset(PoisonA);f.Issue(v);auto b=f.Read();Equal(oracle,b,f.shape,"candidate_vs_oracle");f.Reset(PoisonB);f.Issue(v);auto c=f.Read();Equal(b,c,f.shape,"candidate_AA_dual_poison_complete_write");Equal(oracle,c,f.shape,"candidate_vs_oracle_poisonB");}
static float VerifiedTime(Fixture&f,Variant&v,const std::vector<uint32_t>&oracle,uint32_t poison){f.Reset(poison);cudaEvent_t a,b;Rt(cudaEventCreate(&a));Rt(cudaEventCreate(&b));Rt(cudaEventRecord(a));for(int i=0;i<3;i++)f.Issue(v);Rt(cudaEventRecord(b));Rt(cudaEventSynchronize(b));float ms=0;Rt(cudaEventElapsedTime(&ms,a,b));Rt(cudaEventDestroy(b));Rt(cudaEventDestroy(a));Equal(oracle,f.Read(),f.shape,"timed_half_complete_output");if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("nonpositive/nonfinite event interval");return ms/3;}
int main(int argc,char**argv)try{
 if(argc!=5)throw std::runtime_error("usage: fp4-gemm-pipeline-bench BUILD OUT --strict|--bench|--resources-only VARIANT1..4");
 fs::path build=fs::weakly_canonical(argv[1]);Output=fs::weakly_canonical(argv[2]);std::string mode=argv[3];int id=std::stoi(argv[4]);auto relative=Output.lexically_relative(build);
 if(relative.empty()||*relative.begin()==".."||id<1||id>4||(mode!="--strict"&&mode!="--bench"&&mode!="--resources-only"))throw std::runtime_error("invalid mode/arm/output");fs::create_directories(Output);
 {std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"quality_pass\":false,\"status\":\"GATES_IN_PROGRESS\"}\n";}
 Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));cudaDeviceProp dev{};Rt(cudaGetDeviceProperties(&dev,0));int driver=0,runtime=0;Rt(cudaDriverGetVersion(&driver));Rt(cudaRuntimeGetVersion(&runtime));
 std::printf("{\"event\":\"device\",\"name\":%s,\"cc_major\":%d,\"cc_minor\":%d,\"sms\":%d,\"shared_per_sm\":%zu,\"shared_per_cta_optin\":%zu,\"driver\":%d,\"runtime\":%d}\n",Quote(dev.name).c_str(),dev.major,dev.minor,dev.multiProcessorCount,dev.sharedMemPerMultiprocessor,dev.sharedMemPerBlockOptin,driver,runtime);
 Module module(build,id);Variant&v=Variants[id];size_t strict=0,timed=0,events=0;
 if(mode!="--resources-only"){
  for(const Shape&s:StrictPlan){Fixture f(s);auto oracle=Oracle(f);Strict(f,v,oracle);strict++;std::printf("{\"event\":\"strict_case_pass\",\"variant\":%d,\"shape\":%s,\"swiglu\":%s,\"output_words\":%zu,\"full_output_bits\":true,\"complete_write_dual_poison\":true,\"oracle_AA\":true,\"candidate_AA\":true,\"guards\":true,\"all_inputs_unchanged\":true}\n",id,Quote(PipelineName(s)).c_str(),s.swiglu?"true":"false",f.count);}
  if(mode=="--bench")for(const Shape&s:StrictPlan)if(s.timed){
   if(s.rows<512)throw std::runtime_error("timing is restricted to real prefill shapes");Fixture f(s);auto oracle=Oracle(f);Strict(f,v,oracle);
   // No candidate-specific prepack exists. Every issue is one complete original GEMM/epilogue.
   for(int i=0;i<2;i++){f.Issue(Variants[0]);f.Issue(v);}Rt(cudaDeviceSynchronize());
   for(int round=0;round<5;round++){float a=VerifiedTime(f,Variants[0],oracle,PoisonA),b=VerifiedTime(f,v,oracle,PoisonB),c=VerifiedTime(f,v,oracle,PoisonA),d=VerifiedTime(f,Variants[0],oracle,PoisonB);events++;
    std::printf("{\"event\":\"ABBA_gpu_ms\",\"variant\":%d,\"shape\":%s,\"swiglu\":%s,\"rows\":%d,\"out_dim\":%d,\"k\":%d,\"round\":%d,\"iterations_per_sample\":3,\"fullbits_each_half\":true,\"all_new_work_included\":true,\"A1_oracle_ms\":%.9g,\"B1_candidate_ms\":%.9g,\"B2_candidate_ms\":%.9g,\"A2_oracle_ms\":%.9g}\n",id,Quote(PipelineName(s)).c_str(),s.swiglu?"true":"false",s.rows,s.out_dim,s.k,round,a,b,c,d);
   }timed++;
  }
 }
 std::ofstream f(Output/"result.json");f<<"{\"complete\":true,\"quality_pass\":"<<(mode=="--resources-only"?"false":"true")<<",\"status\":"<<Quote(mode=="--resources-only"?"DRIVER_LAUNCHABILITY_ONLY_FULLBITS_NOT_RUN":"FULLBITS_GUARDS_COVERAGE_AA_PASS")<<",\"variant\":"<<id<<",\"split_k\":1,\"strict_cases\":"<<strict<<",\"timed_shapes\":"<<timed<<",\"timing_events\":"<<events<<",\"rounds\":5,\"iterations_per_sample\":3,\"production_integrated\":false,\"hardware_counters\":\"NOT_RUN\"}\n";return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());if(!Output.empty()){std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"quality_pass\":false,\"status\":\"REJECTED_NOT_VALIDATED\",\"error\":"<<Quote(e.what())<<"}\n";}return 1;}
