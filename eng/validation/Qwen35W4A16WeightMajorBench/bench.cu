// Owned, isolated driver runner. Numerical oracle is BF16(x) times exact W in FP64.
// Original A8 is a different numerical contract, used only as a timed performance control.
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "checks.h"
namespace fs=std::filesystem;
struct Shape {int k,m,n,pattern;float alpha;bool timed;};
#include "plan.h"
static fs::path Output;static int Sms;
constexpr size_t GB=256;constexpr uint32_t G=0x56565656u,P0=0x4f123456u,P1=0xce654321u;
void Rt(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void Cu(CUresult e){if(e!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(e,&s);throw std::runtime_error(s?s:"driver error");}}
std::string Quote(const std::string& s){std::string o="\"";for(char c:s){if(c=='\\'||c=='\"')o+='\\';if(c=='\n'){o+="\\n";continue;}o+=c;}return o+'\"';}
std::string Num(double n){char b[64];if(!std::isfinite(n))return "null";std::snprintf(b,sizeof(b),"%.17g",n);return b;}
uint32_t Hash(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return uint32_t(x^(x>>31));}
std::string Name(Shape s){uint32_t a;std::memcpy(&a,&s.alpha,4);return "k"+std::to_string(s.k)+"-m"+std::to_string(s.m)+"-n"+std::to_string(s.n)+"-p"+std::to_string(s.pattern)+"-a"+std::to_string(a);}
template<class T>void Save(const fs::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("evidence write failed");}
struct Buffer {
 void* base=nullptr;size_t bytes;Buffer(size_t n):bytes(n){Rt(cudaMalloc(&base,n+2*GB));Rt(cudaMemset(base,0x56,n+2*GB));}Buffer(const Buffer&)=delete;~Buffer(){if(base)cudaFree(base);}
 void* Data()const{return (uint8_t*)base+GB;}void Upload(const void* p){Rt(cudaMemcpy(Data(),p,bytes,cudaMemcpyHostToDevice));}
 template<class T>std::vector<T> Read()const{std::vector<T> v(bytes/sizeof(T));Rt(cudaMemcpy(v.data(),Data(),bytes,cudaMemcpyDeviceToHost));return v;}
 void Poison(uint32_t p){if(bytes%4)throw std::runtime_error("poison alignment");std::vector<uint32_t> v(bytes/4,p);Upload(v.data());}
 void Guard()const{std::array<uint32_t,64>a{},b{};Rt(cudaMemcpy(a.data(),base,GB,cudaMemcpyDeviceToHost));Rt(cudaMemcpy(b.data(),(uint8_t*)Data()+bytes,GB,cudaMemcpyDeviceToHost));RequireGuard(a.data(),b.data(),64,G);}
};
struct Module {
 CUmodule mod=nullptr;CUfunction pack,skinny,combine,ref,quant,sweep,old[9]{};
 explicit Module(fs::path build){Cu(cuModuleLoad(&mod,(build/"kernels.cubin").c_str()));
  auto get=[&](CUfunction& f,const char* n){Cu(cuModuleGetFunction(&f,mod,n));};get(pack,"ts_sa16_pack");get(skinny,"ts_wm16_weightmajor");get(combine,"ts_sa16_combine");get(ref,"ts_sa16_reference_f64");get(quant,"ts_q35_quantize_q81_f32");get(sweep,"ts_sa16_cache_sweep");
  for(int n=1;n<=8;n++)get(old[n],("ts_q35_gemv_nvfp4_"+std::to_string(n)).c_str());
  int regs=0,local=0,shared=0,maxblock=0,active=0,binary=0;Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,skinny));Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,skinny));Cu(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,skinny));Cu(cuFuncGetAttribute(&maxblock,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,skinny));Cu(cuFuncGetAttribute(&binary,CU_FUNC_ATTRIBUTE_BINARY_VERSION,skinny));Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,skinny,128,39936));
  printf("{\"event\":\"driver_resources\",\"registers\":%d,\"local_bytes\":%d,\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":39936,\"actual_block\":128,\"active_ctas\":%d,\"max_threads\":%d,\"binary\":%d}\n",regs,local,shared,active,maxblock,binary);
  if(regs<=0||maxblock<128||active<1)throw std::runtime_error("actual driver cannot launch candidate; no register/2CTA hypothesis gate");
 }~Module(){if(mod)cuModuleUnload(mod);}
 void Launch(CUfunction f,unsigned grid,unsigned block,void** args,unsigned z=1,unsigned smem=0){Cu(cuLaunchKernel(f,grid,1,z,block,1,1,smem,nullptr,args,nullptr));}
};
int Split(Shape a){if(a.k<=0||a.k%128||a.m<=0)throw std::runtime_error("invalid split geometry");int z=1,total=a.k/128;while(((a.m+63)/64)*z<2*Sms&&total/(z*2)>=2)z*=2;return z;}
// SA16_HOST_EVICTION_CONTROL_BEGIN
struct CacheSweep {
 Module& m;bool enabled;size_t l2Bytes=0,bytes=0,words=0,workers=0;unsigned blocks=0;size_t calls=0;
 std::unique_ptr<Buffer> data,sums;
 CacheSweep(Module& module,bool on):m(module),enabled(on) {
  CUdevice device;int actual=0;Cu(cuDeviceGet(&device,0));Cu(cuDeviceGetAttribute(&actual,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,device));
  if(actual>0)l2Bytes=(size_t)actual;
  if(on){
   if(l2Bytes==0||l2Bytes>(SIZE_MAX-255)/2)throw std::runtime_error("actual driver L2 size unavailable/overflow; no eviction control");
   bytes=((2*l2Bytes+255)/256)*256;words=bytes/4;blocks=unsigned(Sms)*4;workers=size_t(blocks)*256;
   if(blocks==0||workers==0)throw std::runtime_error("invalid sweep launch geometry");
   data=std::make_unique<Buffer>(bytes);sums=std::make_unique<Buffer>(workers*4);Rt(cudaMemset(data->Data(),1,bytes));Rt(cudaDeviceSynchronize());
  }
  printf("{\"event\":\"cache_control\",\"mode\":\"%s\",\"driver_l2_bytes\":%zu,\"sweep_bytes\":%zu,\"blocks\":%u,\"threads\":256,\"constant_word\":16843009,\"method\":\"%s\",\"physical_cold_guaranteed\":false}\n",on?"eviction-control":"warm",l2Bytes,bytes,blocks,on?"volatile_asm_ld_global_cg_u32":"disabled");
 }
 void Run(const Shape& shape,bool pipeline,int round,const char* half) {
  if(!enabled)return;sums->Poison(P0);void*d=data->Data();void*c=sums->Data();uint64_t n=words;void*args[]={&d,&n,&c};
  m.Launch(m.sweep,blocks,256,args);Rt(cudaDeviceSynchronize());auto values=sums->Read<uint32_t>();data->Guard();sums->Guard();
  uint64_t observed=0,expected=0;
  for(size_t i=0;i<workers;i++) {uint64_t reads=i<words?1+(words-1-i)/workers:0;uint32_t want=(uint32_t)(reads*uint64_t(16843009));if(values[i]!=want)throw std::runtime_error("full-read sweep checksum mismatch");observed+=values[i];expected+=want;}
  calls++;
  printf("{\"event\":\"cache_sweep\",\"shape\":%s,\"scope\":\"%s\",\"round\":%d,\"half\":\"%s\",\"mode\":\"eviction-control\",\"driver_l2_bytes\":%zu,\"sweep_bytes\":%zu,\"checksum\":%llu,\"expected_checksum\":%llu,\"checksum_words\":%zu,\"all_checksums_equal\":true,\"synchronized_before_event\":true,\"physical_cold_guaranteed\":false}\n",Quote(Name(shape)).c_str(),pipeline?"full_pack_kernel_combine":"matrix_kernel_only",round,half,l2Bytes,bytes,(unsigned long long)observed,(unsigned long long)expected,workers);
 }
};
// SA16_HOST_EVICTION_CONTROL_END
struct Fixture {
 Shape s;size_t count;int splits=1,tiles;std::vector<float> hostX;Buffer x,codes,scales,packed,partial,out,reference,absolute,q,qd;
 Fixture(Shape a):s(a),count(size_t(a.n)*a.m),splits(Split(a)),tiles((a.k/128+splits-1)/splits),hostX(size_t(a.n)*a.k),x(hostX.size()*4),codes(size_t(a.m)*a.k/2),scales(size_t(a.m)*a.k/16),packed(size_t(16)*a.k*2),partial(size_t(splits)*a.n*a.m*4),out(count*4),reference(count*8),absolute(count*8),q(size_t(a.n)*a.k),qd(size_t(a.n)*a.k/32*4) {
  if(a.k<=0||a.k%128||a.m<=0||(a.n!=1&&a.n!=2&&a.n!=4&&a.n!=8&&a.n!=16))throw std::runtime_error("unsupported fixture");
  if(splits>256)throw std::runtime_error("split capacity exceeded");
  std::vector<uint8_t>w(codes.bytes),sc(scales.bytes);uint64_t seed=0x20261102ull+a.k*37ull+a.m*19ull+a.n*7ull+a.pattern*101ull;
  for(size_t i=0;i<w.size();i++)w[i]=uint8_t(Hash(seed+i));
  const uint8_t boundary[]={0,1,7,8,31,48,63,95,118,126,127,128,129,254,255};
  for(size_t i=0;i<sc.size();i++)sc[i]=a.pattern==3?boundary[i%15]:a.pattern==6?uint8_t(40+Hash(seed+0x10000000+i)%29):uint8_t(Hash(seed+0x10000000+i));
  if(a.pattern==1){std::fill(w.begin(),w.end(),0);for(int r=0;r<a.m;r++)for(int g=0;g<a.k/16;g++){int j=(r+g)%16;w[size_t(r)*a.k/2+g*8+j%8]|=uint8_t(7u<<((j>=8)?4:0));sc[size_t(r)*a.k/16+g]=uint8_t(14+(r+g)%100);}}
  if(a.pattern==2)for(size_t i=0;i<w.size();i++)w[i]=(i&1)?0xf7:0x7f;
  if(a.pattern==3)for(size_t i=0;i<w.size();i++)w[i]=uint8_t(i*17u);
  const float tiny[]={0.0f,-0.0f,0x1p-120f,-0x1p-100f,0x1p-30f,1.0f,-1.0f,0x1p20f};
  for(size_t i=0;i<hostX.size();i++){uint32_t h=Hash(seed+0x20000000+i);hostX[i]=a.pattern==4?tiny[i%8]:a.pattern==1?float(3*(i/a.k)+int(i%a.k)%71-35)/128.0f:float(int(h%20001)-10000)/8192.0f;}
  x.Upload(hostX.data());codes.Upload(w.data());scales.Upload(sc.data());
 }
 void Pack(Module& m,bool bad=false){void* xp=x.Data();void* p=packed.Data();int n=s.n,k=s.k,b=bad;void*args[]={&xp,&n,&k,&p,&b};m.Launch(m.pack,(16*k+255)/256,256,args);}
 void CheckPack(){auto got=packed.Read<uint16_t>();for(int r=0;r<16;r++)for(int l=0;l<s.k;l++){uint16_t expected=HostBf16(r<s.n?hostX[size_t(r)*s.k+HostPerm(l)]:0);if(got[size_t(r)*s.k+l]!=expected)throw std::runtime_error("full BF16 pack/pad mismatch");}packed.Guard();}
 void PrepareA(Module& m){void* xp=x.Data();void* qp=q.Data();void* dp=qd.Data();int k=s.k,n=s.n;void* args[]={&xp,&qp,&dp,&k,&n};m.Launch(m.quant,(n*(k/32)+3)/4,128,args);}
 void CandidateRaw(Module& m){void*xp=packed.Data();void*w=codes.Data();void*sc=scales.Data();void*y=splits==1?out.Data():partial.Data();int rows=s.m,k=s.k,n=s.n,t=tiles;float alpha=s.alpha;void*args[]={&xp,&w,&sc,&rows,&k,&n,&t,&alpha,&y};m.Launch(m.skinny,(rows+63)/64,128,args,splits,39936);}
 void Finish(Module& m){if(splits>1){void*p=partial.Data();void*y=out.Data();int sp=splits;long long ct=count;float a=s.alpha;void*args[]={&p,&sp,&ct,&a,&y};m.Launch(m.combine,(count+255)/256,256,args);}}
 void OldRaw(Module& m){for(int r=0;r<s.n;r+=8){int n=std::min(8,s.n-r),rows=s.m,k=s.k,rpw=n==1?1:n<=3?2:4;float a=s.alpha;void*w=codes.Data();void*sc=scales.Data();void*xp=(uint8_t*)q.Data()+size_t(r)*k;void*dp=(uint8_t*)qd.Data()+size_t(r)*(k/32)*4;void*y=(uint8_t*)out.Data()+size_t(r)*s.m*4;void*args[]={&w,&sc,&rows,&k,&xp,&dp,&a,&y};m.Launch(m.old[n],(rows+4*rpw-1)/(4*rpw),128,args);}}
 void Issue(Module& m,bool candidate,bool pipeline){if(candidate){if(pipeline)Pack(m);CandidateRaw(m);if(pipeline)Finish(m);}else{if(pipeline)PrepareA(m);OldRaw(m);}}
 void Reference(Module& m){void*xp=x.Data();void*w=codes.Data();void*sc=scales.Data();int rows=s.m,k=s.k,n=s.n;float a=s.alpha;void*y=reference.Data();void*ab=absolute.Data();void*args[]={&xp,&w,&sc,&rows,&k,&n,&a,&y,&ab};m.Launch(m.ref,(count+3)/4,128,args);}
 std::vector<float> Read(){Rt(cudaDeviceSynchronize());for(Buffer* p:{&x,&codes,&scales,&packed,&partial,&out,&reference,&absolute,&q,&qd})p->Guard();return out.Read<float>();}
};
void Bits(const std::vector<float>& a,const std::vector<float>& b){if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*4))throw std::runtime_error("full output A/A or post-timing bits mismatch");}
struct Expected {std::vector<float>a,b;};
Expected Strict(Fixture& f,Module& m,bool mutation=false,const char* phase="strict"){
 auto dir=Output/(std::string(phase)+"-"+Name(f.s));if(!fs::create_directory(dir))throw std::runtime_error("fresh strict evidence directory required");f.Pack(m);f.CheckPack();f.reference.Poison(P0);f.absolute.Poison(P1);f.Reference(m);Rt(cudaDeviceSynchronize());auto ref=f.reference.Read<double>(),ab=f.absolute.Read<double>();Save(dir/"reference.f64",ref);Save(dir/"reference-absolute.f64",ab);
 f.reference.Poison(P1);f.absolute.Poison(P0);f.Reference(m);Rt(cudaDeviceSynchronize());auto refAA=f.reference.Read<double>(),abAA=f.absolute.Read<double>();if(std::memcmp(ref.data(),refAA.data(),ref.size()*8)||std::memcmp(ab.data(),abAA.data(),ab.size()*8))throw std::runtime_error("reference A/A changed");
 f.out.Poison(P0);f.Issue(m,false,true);auto a=f.Read();f.out.Poison(P1);f.Issue(m,false,true);Bits(a,f.Read());Save(dir/"legacy-A8.f32",a);
 f.out.Poison(P0);f.partial.Poison(P0);f.Issue(m,true,true);auto b=f.Read();Save(dir/"candidate-BF16.f32",b);auto numeric=CheckNumeric(b,ref,ab,f.s.k,f.splits);
 f.out.Poison(P1);f.partial.Poison(P1);f.Issue(m,true,true);auto aa=f.Read();Save(dir/"candidate-AA.f32",aa);Bits(b,aa);
 std::ofstream evidence(dir/"numeric.json");evidence<<"{\"passed\":"<<(numeric.pass?"true":"false")<<",\"bad_outputs\":"<<numeric.bad<<",\"relative_l2\":"<<Num(numeric.relativeL2)<<",\"max_absolute_bound_ratio\":"<<Num(numeric.maxRatio)<<",\"splits\":"<<f.splits<<",\"full_output_count\":"<<f.count<<",\"candidate_AA_bits\":true,\"dual_poison\":true,\"old_A8_equivalence_claim\":false}\n";evidence.close();
 if(!numeric.pass)throw std::runtime_error("independent FP64 numerical gate failed: "+Name(f.s));
 if(mutation){f.Pack(m,true);f.out.Poison(P1);f.CandidateRaw(m);f.Finish(m);auto wrong=f.Read();Save(dir/"mutation-wrong-permutation.f32",wrong);if(CheckNumeric(wrong,ref,ab,f.s.k,f.splits).pass)throw std::runtime_error("wrong swizzle mutation escaped gate");auto scaled=b;for(float&v:scaled)v*=1.01f;if(CheckNumeric(scaled,ref,ab,f.s.k,f.splits).pass)throw std::runtime_error("scale1.01 mutation escaped gate");f.Pack(m);}
 printf("{\"event\":\"strict_case_pass\",\"shape\":%s,\"n\":%d,\"splits\":%d,\"outputs\":%zu,\"relative_l2\":%.9g,\"full_output_saved\":true,\"AA\":true,\"guards\":true,\"mutation_checked\":%s}\n",Quote(Name(f.s)).c_str(),f.s.n,f.splits,f.count,numeric.relativeL2,mutation?"true":"false");return{a,b};
}
float Time(Fixture& f,Module& m,bool candidate,bool pipeline,int iterations,const Expected& expected,CacheSweep& sweep,const char* half,int round){
 f.out.Poison(P0);f.partial.Poison(P1);cudaEvent_t start,end;Rt(cudaEventCreate(&start));Rt(cudaEventCreate(&end));sweep.Run(f.s,pipeline,round,half);Rt(cudaEventRecord(start));for(int i=0;i<iterations;i++)f.Issue(m,candidate,pipeline);Rt(cudaEventRecord(end));Rt(cudaEventSynchronize(end));float ms=0;Rt(cudaEventElapsedTime(&ms,start,end));Rt(cudaEventDestroy(start));Rt(cudaEventDestroy(end));
 if(!pipeline&&candidate)f.Finish(m);Bits(candidate?expected.b:expected.a,f.Read());if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("nonpositive/nonfinite CUDA event timing");return ms/iterations;
}
int main(int argc,char**argv)try{
 if(argc<4)throw std::runtime_error("usage: weightmajor-bench BUILD OUT --strict|--bench [--rounds 5 --iterations 3]");fs::path build=fs::canonical(argv[1]);Output=fs::canonical(argv[2]);auto rel=Output.lexically_relative(build);if(rel.empty()||*rel.begin()=="..")throw std::runtime_error("fresh output must be below build");std::string mode=argv[3];int rounds=5,iterations=3;bool cacheSweep=false,explicitIterations=false;
 for(int i=4;i<argc;i++){std::string opt=argv[i];if(opt=="--cache-sweep"){cacheSweep=true;continue;}if(i+1>=argc)throw std::runtime_error("missing option");if(opt=="--rounds")rounds=std::stoi(argv[++i]);else if(opt=="--iterations"){iterations=std::stoi(argv[++i]);explicitIterations=true;}else throw std::runtime_error("unknown option");}
 if(cacheSweep&&!explicitIterations)iterations=1;
 if(cacheSweep&&(mode!="--bench"||iterations!=1))throw std::runtime_error("--cache-sweep requires --bench and iterations1");
 if((mode!="--strict"&&mode!="--bench")||rounds<1||iterations<1)throw std::runtime_error("invalid configuration");
 {std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"quality_passed\":false,\"status\":\"RUNNING\"}\n";}
 Rt(cudaSetDevice(0));Rt(cudaFree(nullptr));Cu(cuInit(0));cudaDeviceProp prop{};Rt(cudaGetDeviceProperties(&prop,0));Sms=prop.multiProcessorCount;
 printf("{\"event\":\"device\",\"name\":%s,\"sms\":%d,\"cc_major\":%d,\"cc_minor\":%d}\n",Quote(prop.name).c_str(),Sms,prop.major,prop.minor);if(Sms<1)throw std::runtime_error("invalid actual SM count");Module module(build);CacheSweep sweep(module,cacheSweep);size_t strict=0,timed=0;
 for(Shape s:StrictPlan){Fixture f(s);Strict(f,module,s.k==128&&s.m==17&&s.n==8&&s.pattern==1);strict++;}
 if(mode=="--bench")for(Shape s:StrictPlan)if(s.timed){Fixture f(s);Expected expected=Strict(f,module,false,"timed-regate");f.Pack(module);f.PrepareA(module);for(bool pipeline:{false,true}){for(int w=0;w<2;w++){f.Issue(module,false,pipeline);f.Issue(module,true,pipeline);}Rt(cudaDeviceSynchronize());for(int r=0;r<rounds;r++){float a=Time(f,module,false,pipeline,iterations,expected,sweep,"A1",r),b=Time(f,module,true,pipeline,iterations,expected,sweep,"B1",r),c=Time(f,module,true,pipeline,iterations,expected,sweep,"B2",r),d=Time(f,module,false,pipeline,iterations,expected,sweep,"A2",r);printf("{\"event\":\"ABBA_gpu_ms\",\"shape\":%s,\"scope\":\"%s\",\"n\":%d,\"splits\":%d,\"round\":%d,\"cache_mode\":\"%s\",\"iterations_per_sample\":%d,\"A1_legacy_ms\":%.9g,\"B1_bf16_ms\":%.9g,\"B2_bf16_ms\":%.9g,\"A2_legacy_ms\":%.9g,\"full_output_postsample_bits\":true}\n",Quote(Name(s)).c_str(),pipeline?"full_pack_kernel_combine":"matrix_kernel_only",s.n,f.splits,r,cacheSweep?"eviction-control":"warm",iterations,a,b,c,d);}}timed++;}
 std::ofstream f(Output/"result.json");f<<"{\"schema\":\"q35-w4a16-weightmajor-result/1\",\"complete\":true,\"quality_passed\":true,\"status\":\"FP64_NUMERIC_AA_GUARDS_PASS\",\"strict_cases\":"<<strict<<",\"timed_shapes\":"<<timed<<",\"rounds\":"<<rounds<<",\"iterations\":"<<iterations<<",\"cache_mode\":\""<<(cacheSweep?"eviction-control":"warm")<<"\",\"cache_sweep_calls\":"<<sweep.calls<<",\"driver_l2_bytes\":"<<sweep.l2Bytes<<",\"sweep_bytes\":"<<sweep.bytes<<",\"old_A8_equivalence_claim\":false,\"production_integrated\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());if(!Output.empty()){std::ofstream f(Output/"result.json");f<<"{\"complete\":false,\"quality_passed\":false,\"status\":\"ERROR_OR_NUMERIC_FAIL\",\"error\":"<<Quote(e.what())<<"}\n";}return 1;}
