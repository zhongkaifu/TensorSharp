#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
namespace Q6Check {
inline size_t RawBytes(int m,int k){if(m<1||k<256||k%256)throw std::runtime_error("invalid Q6 shape");return size_t(m)*(k/256)*210;}
inline size_t PackedBytes(int m,int k,bool bytes=false,bool aligned=false){RawBytes(m,k);if(bytes&&aligned)throw std::runtime_error("mutually exclusive layout");return size_t(m)*(k/256)*(bytes?210:aligned?212:274);}
// Direct per-element specification, independent of the four-byte original GPU decode.
inline int Code(const uint8_t* block,int j) {
 const int half=j/128,segment=j/32%4,lane=j%32;
 int low=(block[half*64+(segment%2)*32+lane]>>(segment>=2?4:0))&15;
 int high=(block[128+half*32+lane]>>(segment*2))&3;
 return (low+16*high)-32;
}
inline std::vector<uint8_t> Pack(const std::vector<uint8_t>& raw,int m,int k) {
 if(raw.size()!=RawBytes(m,k))throw std::runtime_error("raw size mismatch");
 std::vector<uint8_t> out(PackedBytes(m,k));const size_t count=size_t(m)*k;
 for(int r=0;r<m;r++)for(int sb=0;sb<k/256;sb++) {
  const uint8_t* b=raw.data()+(size_t(r)*(k/256)+sb)*210;
  for(int j=0;j<256;j++)out[size_t(r)*k+sb*256+j]=uint8_t(Code(b,j));
  std::memcpy(out.data()+count+size_t(r)*(k/16)+sb*16,b+192,16);
  std::memcpy(out.data()+count+size_t(m)*(k/16)+(size_t(r)*(k/256)+sb)*2,b+208,2);
 }
 return out;
}
inline std::vector<uint8_t> BytePlanes(const std::vector<uint8_t>& raw,int m,int k) {
 if(raw.size()!=RawBytes(m,k))throw std::runtime_error("raw size mismatch");
 const size_t blocks=size_t(m)*(k/256);std::vector<uint8_t> out(raw.size());
 for(size_t block=0;block<blocks;block++) {
  std::memcpy(out.data()+block*128,raw.data()+block*210,128);
  std::memcpy(out.data()+blocks*128+block*64,raw.data()+block*210+128,64);
  std::memcpy(out.data()+blocks*192+block*16,raw.data()+block*210+192,16);
  std::memcpy(out.data()+blocks*208+block*2,raw.data()+block*210+208,2);
 }
 return out;
}
inline std::vector<uint8_t> AlignedBlocks(const std::vector<uint8_t>& raw,int m,int k) {
 if(raw.size()!=RawBytes(m,k))throw std::runtime_error("raw size mismatch");
 const size_t blocks=size_t(m)*(k/256);std::vector<uint8_t> out(blocks*212,0);
 for(size_t b=0;b<blocks;b++)std::memcpy(out.data()+b*212,raw.data()+b*210,210);
 return out;
}
inline void Verify(const std::vector<uint8_t>& raw,const std::vector<uint8_t>& packed,int m,int k,bool bytes=false,bool aligned=false) {
 if(bytes&&aligned)throw std::runtime_error("mutually exclusive layout");
 if((bytes?BytePlanes(raw,m,k):aligned?AlignedBlocks(raw,m,k):Pack(raw,m,k))!=packed)throw std::runtime_error("whole packed bytes differ from independent CPU decoder");
}
}
