// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Fast fixed-shape screening only. No FP64, quantizer, model or adoption claim.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#define R(c) do { auto e=(c); if(e!=cudaSuccess){std::cerr<<#c<<": "<<cudaGetErrorString(e)<<"\n";return 1;} }while(0)
#define D(c) do { auto e=(c); if(e!=CUDA_SUCCESS){std::cerr<<#c<<": "<<int(e)<<"\n";return 1;} }while(0)
int main(int argc,char** argv){
 if(argc!=4&&argc!=5&&argc!=6&&argc!=7)return 2;
 const bool legacy8=argc==7;if(legacy8&&std::string(argv[6])!="--legacy-bn8")return 2;
 const int matrices=argc>=5?std::atoi(argv[4]):1;if(matrices!=1&&matrices!=6)return 2;
 const std::string shape=argc>=6?argv[5]:"gdn";
 int n=16384,k=5120,splits=2;
 if(shape=="gate-up"){n=34816;splits=1;}else if(shape=="down"){n=5120;k=17408;splits=8;}else if(shape=="ssm-out"){n=5120;k=6144;splits=8;}else if(shape=="attn-qkv"){n=14336;splits=2;}else if(shape!="gdn")return 2;
 constexpr int m=8,guard=256;const int tiles=n/128;
 const size_t output=size_t(n)*m*4,part=output*splits;
 R(cudaSetDevice(0)); R(cudaFree(nullptr));
 CUmodule modules[2]; CUfunction kernels[2];
 for(int a=0;a<2;a++){D(cuModuleLoad(&modules[a],argv[a+1])); D(cuModuleGetFunction(&kernels[a],modules[a],a||legacy8?"ts_q35_gemm_w4a4_8":"ts_q35_gemm_w4a4_32")); D(cuFuncSetAttribute(kernels[a],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,a||legacy8?43520:76800));}
 auto alloc=[](size_t size){void* p=nullptr;if(cudaMalloc(&p,size)!=cudaSuccess)std::abort();return p;};
 auto upload=[&](const void* h,size_t size){void* p=alloc(size);if(cudaMemcpy(p,h,size,cudaMemcpyHostToDevice)!=cudaSuccess)std::abort();return p;};
 std::vector<uint8_t> wc(size_t(n)*k/2),ws(size_t(n)*k/16,0x38),xc(size_t(m)*k/2),xs(size_t(m)*k/16,0x38);
 uint32_t rng=1234;for(auto&v:wc){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;v=uint8_t(rng);}for(size_t i=0;i<xc.size();i++)xc[i]=uint8_t(i*37+11);
 float scales[m];for(auto&v:scales)v=1;
 void *wcodes=upload(wc.data(),wc.size()),*wscales=upload(ws.data(),ws.size()),*xcodes=upload(xc.data(),xc.size()),*xscales=upload(xs.data(),xs.size()),*rowscale=upload(scales,sizeof scales);
 std::vector<void*> weightCodes{wcodes},weightScales{wscales};for(int i=1;i<matrices;i++){weightCodes.push_back(upload(wc.data(),wc.size()));weightScales.push_back(upload(ws.data(),ws.size()));}
 void *ya[2],*pa[2],*ca[2],*y[2],*partial[2],*counter[2];
 for(int a=0;a<2;a++){ya[a]=alloc(output+2*guard);pa[a]=alloc(part+2*guard);ca[a]=alloc(tiles*4+2*guard);R(cudaMemset(ya[a],0xa5,output+2*guard));R(cudaMemset(pa[a],0xa5,part+2*guard));R(cudaMemset(ca[a],0xa5,tiles*4+2*guard));y[a]=(char*)ya[a]+guard;partial[a]=(char*)pa[a]+guard;counter[a]=(char*)ca[a]+guard;R(cudaMemset(counter[a],0,tiles*4));}
 cudaStream_t stream;R(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));float alpha=1;
 auto launch=[&](int a,int index){int nn=n,kk=k,mm=m;void *cw=weightCodes[index],*sw=weightScales[index];void* args[]={&cw,&sw,&nn,&kk,&xcodes,&xscales,&rowscale,&mm,&alpha,&y[a],&partial[a],&counter[a]};return cuLaunchKernel(kernels[a],tiles,1,splits,256,1,1,a||legacy8?43520:76800,(CUstream)stream,args,nullptr);};
 auto read=[](void*p,size_t size){std::vector<uint8_t> h(size);if(cudaMemcpy(h.data(),p,size,cudaMemcpyDeviceToHost)!=cudaSuccess)std::abort();return h;};
 auto check=[&](){if(read(y[0],output)!=read(y[1],output)||read(partial[0],part)!=read(partial[1],part))return false;for(int a=0;a<2;a++){for(auto b:read(counter[a],tiles*4))if(b)return false;for(auto p:{std::pair<void*,size_t>{ya[a],output},{pa[a],part},{ca[a],tiles*4}}){auto h=read(p.first,p.second+2*guard);for(int i=0;i<guard;i++)if(h[i]!=0xa5||h[h.size()-guard+i]!=0xa5)return false;}auto h=read(y[a],output);for(size_t i=0;i<h.size();i+=4){float v;std::memcpy(&v,h.data()+i,4);if(!std::isfinite(v))return false;}}return true;};
 for(int i=0;i<matrices;i++){for(int a=0;a<2;a++)D(launch(a,i));R(cudaStreamSynchronize(stream));if(!check())return 3;}
 cudaGraph_t graph[2];cudaGraphExec_t exec[2];
 for(int a=0;a<2;a++){R(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int i=0;i<matrices;i++)D(launch(a,i));R(cudaStreamEndCapture(stream,&graph[a]));size_t nodes=0;R(cudaGraphGetNodes(graph[a],nullptr,&nodes));if(nodes!=size_t(matrices))return 4;R(cudaGraphInstantiateWithFlags(&exec[a],graph[a],0));}
 cudaEvent_t begin,end;R(cudaEventCreate(&begin));R(cudaEventCreate(&end));
 std::vector<float> values[2];std::ofstream log(std::string(argv[3])+"/intervals.jsonl");if(!log)return 6;log.precision(12);
 for(int round=0;round<5;round++)for(int pos=0;pos<4;pos++){int a=(pos==1||pos==2)?1:0;R(cudaGraphLaunch(exec[a],stream));R(cudaStreamSynchronize(stream));R(cudaEventRecord(begin,stream));for(int i=0;i<64;i++)R(cudaGraphLaunch(exec[a],stream));R(cudaEventRecord(end,stream));R(cudaEventSynchronize(end));float ms;R(cudaEventElapsedTime(&ms,begin,end));if(!(ms>0)||!std::isfinite(ms)||!check())return 5;values[a].push_back(ms/64);log<<"{\"round\":"<<round<<",\"position\":"<<pos<<",\"arm\":"<<a<<",\"millisecondsPerMatrixPass\":"<<ms/64<<"}\n";}
 for(int a=0;a<2;a++){auto h=read(y[a],output);std::ofstream f(std::string(argv[3])+(a?"/candidate.bin":"/legacy.bin"),std::ios::binary);f.write((char*)h.data(),h.size());f.flush();if(!f)return 6;}
 log.flush();if(!log)return 6;
 auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return(double(v[4])+v[5])/2;};double old=median(values[0]),candidate=median(values[1]);
 for(int a=0;a<2;a++){R(cudaGraphExecDestroy(exec[a]));R(cudaGraphDestroy(graph[a]));D(cuModuleUnload(modules[a]));for(auto p:{ya[a],pa[a],ca[a]})R(cudaFree(p));}for(auto p:weightCodes)R(cudaFree(p));for(auto p:weightScales)R(cudaFree(p));for(auto p:{xcodes,xscales,rowscale})R(cudaFree(p));R(cudaEventDestroy(begin));R(cudaEventDestroy(end));R(cudaStreamDestroy(stream));
 std::cout.precision(12);std::cout<<"{\"screenPassed\":true,\"legacySchedule\":\""<<(legacy8?"bn8":"bn32")<<"\",\"legacyMs\":"<<old<<",\"candidateMs\":"<<candidate<<",\"coreSpeedup\":"<<old/candidate<<",\"scope\":\"Single-shape raw-payload one or six-address matrix-pass graph; raw output/scratch bytes equal, guards checked (split1 scratch is unused). No quantizer/FP64/DRAM-counter/model/E2E qualification.\"}\n";
}
