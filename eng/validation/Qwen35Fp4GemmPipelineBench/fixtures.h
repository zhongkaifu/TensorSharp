// Host-only input generator shared by the GPU runner and CPU distribution gates.
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
struct Shape {int rows,out_dim,k,pattern;uint32_t alpha;bool timed,swiglu;};
inline uint32_t PipelineHash(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;x^=x>>31;return uint32_t(x);}
inline std::string PipelineName(const Shape&s){return "r"+std::to_string(s.rows)+"-o"+std::to_string(s.out_dim)+"-k"+std::to_string(s.k)+"-p"+std::to_string(s.pattern)+"-a"+std::to_string(s.alpha)+"-f"+std::to_string(s.swiglu);}
struct PipelineInputs {
 std::vector<uint8_t>wc,ws,xc,xs;std::vector<uint32_t>scale;
 explicit PipelineInputs(const Shape&s){
  if(s.rows<=64||s.out_dim<=0||s.k<=0||s.k%256||s.pattern<0||s.pattern>6||(s.swiglu&&s.out_dim%16))throw std::runtime_error("unsupported fixture geometry");
  wc.resize(size_t(s.out_dim)*(s.k/2));ws.resize(size_t(s.out_dim)*(s.k/16));xc.resize(size_t(s.rows)*(s.k/2));xs.resize(size_t(s.rows)*(s.k/16));scale.resize(s.rows);
  uint64_t seed=0x20261102ull+uint64_t(s.rows)*41+uint64_t(s.out_dim)*71+uint64_t(s.k)*137+uint64_t(s.pattern)*19;
  auto codes=[&](std::vector<uint8_t>&v,uint64_t salt){for(size_t i=0;i<v.size();i++)v[i]=(s.pattern==0||s.pattern==6)?uint8_t(PipelineHash(seed+salt+i)):s.pattern==1?uint8_t((i/(s.k/2))&1?0xff:0x77):s.pattern==2?uint8_t((i%4==0)?0x00:(i%4==1)?0x88:(i%4==2)?0x08:0x80):s.pattern==3?uint8_t(((i%size_t(s.k/2))/32)&1?0xff:0x77):s.pattern==4?uint8_t(i%17==0?0x7f:0):uint8_t(i*17u);};
  const uint8_t levels[]={0,1,7,8,15,0x38,0x7d,0x7e};
  auto scales=[&](std::vector<uint8_t>&v,uint64_t salt){for(size_t i=0;i<v.size();i++)v[i]=s.pattern==6?uint8_t(PipelineHash(seed+salt+i)%64):s.pattern==0?uint8_t(PipelineHash(seed+salt+i)%127):s.pattern==1?0x7e:s.pattern==2?0x38:s.pattern==3?uint8_t(i&1?0x38:0x40):s.pattern==4?levels[i%8]:0x7e;};
  codes(wc,0x10000000ull);codes(xc,0x20000000ull);scales(ws,0x30000000ull);scales(xs,0x40000000ull);
  const uint32_t ordinary[]={0,0x80000000u,0x3f800000u,0xbf800000u,0x3f000000u,0xbf400000u,0x3f800001u,0x3a800000u};
  const uint32_t tiny[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u};
  for(size_t i=0;i<scale.size();i++)scale[i]=s.pattern==6?0x39800000u+(PipelineHash(seed+0x50000000ull+i)%0x1000000u):s.pattern==0?0x3f000000u+(PipelineHash(seed+0x50000000ull+i)%0x1000000u):s.pattern==2?uint32_t(i&1?0x80000000u:0):s.pattern==4?tiny[i%8]:s.pattern==5?uint32_t(i&1?0xff7fffffu:0x7f7fffffu):ordinary[i%8];
 }
};
