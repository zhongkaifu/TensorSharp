#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
// Fixed before device measurement. Never adjusted on observed results.
constexpr double SA16_RELATIVE_L2_LIMIT=0.00002;
constexpr double SA16_UNIT_ROUNDOFF=0x1p-24;
struct NumericCheck { bool pass=true;size_t bad=0,first=SIZE_MAX;double maxRatio=0,relativeL2=0; };
inline NumericCheck CheckNumeric(const std::vector<float>& actual,const std::vector<double>& ref,const std::vector<double>& absolute,int k,int splits) {
    if(actual.size()!=ref.size()||ref.size()!=absolute.size()||actual.empty())throw std::runtime_error("numeric geometry differs");
    // Conservative summation error envelope for F32 MMA + fixed-order split reduction.
    // It is an explicit experimental criterion, not a claim about undocumented Tensor Core rounding.
    double t=(2.0*k+2.0*splits+8)*SA16_UNIT_ROUNDOFF,gamma=t/(1-t),err2=0,ref2=0;NumericCheck r;
    for(size_t i=0;i<actual.size();i++) {
        double err=std::abs((double)actual[i]-ref[i]),bound=gamma*absolute[i]+0x1p-140;
        bool ok=std::isfinite(actual[i])&&std::isfinite(ref[i])&&std::isfinite(absolute[i])&&absolute[i]>=0&&err<=bound;
        if(!ok){r.bad++;if(r.first==SIZE_MAX)r.first=i;}
        r.maxRatio=std::max(r.maxRatio,bound>0?err/bound:err==0?0:INFINITY);err2+=err*err;ref2+=ref[i]*ref[i];
    }
    r.relativeL2=ref2>0?std::sqrt(err2/ref2):err2==0?0:INFINITY;
    r.pass=r.bad==0&&std::isfinite(r.relativeL2)&&r.relativeL2<=SA16_RELATIVE_L2_LIMIT;return r;
}
inline uint16_t HostBf16(float value) {uint32_t b;std::memcpy(&b,&value,4);return (uint16_t)((b+0x7fff+((b>>16)&1))>>16);}
inline float HostFloat(uint32_t value) {float x;std::memcpy(&x,&value,4);return x;}
inline int HostPerm(int l) {int s=(l>>4)&1,h=(l>>3)&1,q=(l>>1)&3,j=l&1;return (l&~31)+16*(q>>1)+4*(q&1)+2*s+h+8*j;}

inline void RequireGuard(const uint32_t* before,const uint32_t* after,size_t count,uint32_t expected) {for(size_t i=0;i<count;i++)if(before[i]!=expected||after[i]!=expected)throw std::runtime_error("allocation guard changed");}
