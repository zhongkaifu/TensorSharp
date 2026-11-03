// Copyright (c) Zhongkai Fu. All rights reserved.
// Independent CPU layout/rounding/reference and preregistered error contract.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
static uint32_t FBits(float f){uint32_t v;std::memcpy(&v,&f,4);return v;}
static float FValue(uint32_t v){float f;std::memcpy(&f,&v,4);return f;}
static uint16_t Bf16(float f){
    uint32_t v=FBits(f);if((v&0x7f800000u)==0x7f800000u)throw std::runtime_error("nonfinite FP32 input");
    uint16_t b=uint16_t((v+0x7fffu+((v>>16)&1u))>>16);
    if((b&0x7f80u)==0x7f80u)throw std::runtime_error("nonfinite BF16 cast");
    if((b&0x7f80u)==0&&(b&127u))throw std::runtime_error("BF16 subnormal activation unsupported");
    return b;
}
static float BValue(uint16_t b){return FValue(uint32_t(b)<<16);}
static double HalfValue(uint16_t h){
    int e=(h>>10)&31,m=h&1023;if(e==31)throw std::runtime_error("nonfinite half scale");
    double v=e?std::ldexp(1.0+double(m)/1024,e-15):std::ldexp(double(m),-24);
    return(h&0x8000u)?-v:v;
}
static uint16_t RoundBf16Reference(double x){
    if(!std::isfinite(x))throw std::runtime_error("nonfinite reference");
    // F64→F32 double-rounding is avoided: compare adjacent BF16 exact values.
    float xf=float(x);if(!std::isfinite(xf))throw std::runtime_error("reference outside F32");
    uint32_t bits=FBits(xf);uint16_t center=uint16_t(bits>>16),best=center;
    double distance=std::abs(x-double(BValue(center)));
    for(int d=-2;d<=2;d++){
        int c=int(center)+d;if(c<0||c>65535)continue;uint16_t b=uint16_t(c);
        if((b&0x7f80u)==0x7f80u)continue;double z=std::abs(x-double(BValue(b)));
        if(z<distance||(z==distance&&(b&1u)==0)){distance=z;best=b;}
    }
    if(x==0)return std::signbit(x)?0x8000:0;
    return best;
}
struct W8Layout {
    uint64_t rows,k,kpad,codes,scales_at,groups,total;
    W8Layout(uint64_t r,uint64_t width):rows(r),k(width){
        if(!r||!width||width%32||width>65536||r>1048576)throw std::runtime_error("unsupported/bounded W8 shape");
        kpad=(width+127)/128*128;codes=rows*kpad;scales_at=(codes+255)/256*256;groups=kpad/32;total=scales_at+rows*groups*2;
        if(total>std::numeric_limits<size_t>::max())throw std::runtime_error("W8 size overflow");
    }
    uint64_t Code(uint64_t r,uint64_t c)const{if(r>=rows||c>=kpad)throw std::runtime_error("code index");return r*kpad+c;}
    uint64_t Scale(uint64_t r,uint64_t g)const{if(r>=rows||g>=groups)throw std::runtime_error("scale index");return scales_at+2*(r*groups+g);}
};
static int SignedCode(uint8_t q){return q<128?int(q):int(q)-256;}
static uint16_t ReadHalf(const std::vector<uint8_t>& w,uint64_t at){if(at+2>w.size())throw std::runtime_error("half bounds");return uint16_t(w[at])|(uint16_t(w[at+1])<<8);}
static void WriteHalf(std::vector<uint8_t>& w,uint64_t at,uint16_t h){if(at+2>w.size())throw std::runtime_error("half bounds");w[at]=uint8_t(h);w[at+1]=uint8_t(h>>8);}
static void DecodeLayout(const std::vector<uint8_t>& w,const W8Layout& l,
                         std::vector<int8_t>& codes,std::vector<uint16_t>& scales){
    if(w.size()!=l.total)throw std::runtime_error("layout length");
    codes.resize(l.rows*l.kpad);scales.resize(l.rows*l.groups);
    for(uint64_t r=0;r<l.rows;r++){
        for(uint64_t c=0;c<l.kpad;c++)codes[r*l.kpad+c]=int8_t(SignedCode(w[l.Code(r,c)]));
        for(uint64_t g=0;g<l.groups;g++)scales[r*l.groups+g]=ReadHalf(w,l.Scale(r,g));
    }
}
static void VerifyLayout(const std::vector<uint8_t>& w,const W8Layout& l,
                         const std::vector<int8_t>& codes,const std::vector<uint16_t>& scales){
    std::vector<int8_t> gotCodes;std::vector<uint16_t> gotScales;DecodeLayout(w,l,gotCodes,gotScales);
    if(gotCodes!=codes||gotScales!=scales)throw std::runtime_error("layout code/scale bits mismatch");
}
static double Gamma(uint64_t n,double u){double z=n*u;if(z>=1)throw std::runtime_error("error bound overflow");return z/(1-z);}
static double Allowed(double ref,double sumabs,int k){
    if(!std::isfinite(ref)||!std::isfinite(sumabs)||sumabs<0)throw std::runtime_error("invalid reference bounds");
    uint64_t n=uint64_t(k/32)+64;
    double e=(Gamma(n,std::ldexp(1.0,-24))+Gamma(k,std::ldexp(1.0,-53)))*sumabs+n*std::ldexp(1.0,-149);
    return e+std::ldexp(1.0,-8)*(std::abs(ref)+e)+std::ldexp(1.0,-134);
}
static void CheckOutput(const std::vector<uint16_t>& out,const std::vector<double>& ref,
                        const std::vector<double>& sumabs,int k,bool exact){
    if(out.size()!=ref.size()||ref.size()!=sumabs.size())throw std::runtime_error("output length");
    for(size_t i=0;i<out.size();i++){
        double value=BValue(out[i]);
        if(!std::isfinite(value)||std::abs(value-ref[i])>Allowed(ref[i],sumabs[i],k))
            throw std::runtime_error("preregistered numeric tolerance failed");
        if(exact&&out[i]!=RoundBf16Reference(ref[i]))throw std::runtime_error("exact integer/power2 BF16 bits failed");
    }
}
static void CpuReference(const std::vector<uint8_t>& w,const W8Layout& l,const std::vector<uint16_t>& x,int tokens,
                         std::vector<double>& y,std::vector<double>& sumabs,std::vector<uint16_t>& f32){
    if(w.size()!=l.total||x.size()!=size_t(tokens)*l.k)throw std::runtime_error("CPU reference bounds");
    y.assign(size_t(tokens)*l.rows,0);sumabs.assign(y.size(),0);f32.assign(y.size(),0);
    for(int t=0;t<tokens;t++)for(uint64_t r=0;r<l.rows;r++){
        double z=0,a=0;float acc=0;
        for(uint64_t g=0;g<l.k/32;g++){
            double scale=HalfValue(ReadHalf(w,l.Scale(r,g)));float group=0;
            for(int j=0;j<32;j++){
                int q=SignedCode(w[l.Code(r,g*32+j)]);float v=BValue(x[size_t(t)*l.k+g*32+j]);
                double p=double(q)*scale*double(v);z+=p;a+=std::abs(p);group=std::fma(float(q),v,group);
            }
            acc=std::fma(group,float(scale),acc);
        }
        if(!std::isfinite(z)||!std::isfinite(a)||!std::isfinite(acc))throw std::runtime_error("nonfinite reference result");
        size_t i=size_t(t)*l.rows+r;y[i]=z;sumabs[i]=a;f32[i]=RoundBf16Reference(double(acc));
    }
}
