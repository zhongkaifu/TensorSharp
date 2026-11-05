// TensorSharp-owned experimental W4A16 kernel; not production dispatch.
// Algorithm inspiration only: Oceania2018/velox-inferer b67f7d4565f5335988c91869cf0590b1e05998cd,
// crates/velox-variant/kernels/rtx5090_skinny16.cu (exact upstream path recorded by build manifest).
// No Velox implementation is included. TS's GGUF nibble order and UE4M3 semantics apply.
constexpr int SA16_ROWS=64, SA16_K=128, SA16_STAGES=4, SA16_THREADS=128, SA16_PAD=136;
constexpr int SA16_SHARED=SA16_STAGES*(16*SA16_PAD*2+64*80+64*8);
struct SA16Storage { __nv_bfloat16 x[4][16*136]; uint8_t w[4][64*80]; uint8_t s[4][64*8]; };
static_assert(sizeof(SA16Storage)==SA16_SHARED,"shared geometry");
__host__ __device__ int sa16_permute(int l) {
    int s=(l>>4)&1,h=(l>>3)&1,q=(l>>1)&3,j=l&1;
    return (l&~31)+16*(q>>1)+4*(q&1)+2*s+h+8*j;
}
__device__ uint32_t sa16_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ void sa16_cp16(void* d,const void* s,bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(sa16_smem(d)),"l"(s),"r"(ok?16:0));
}
__device__ void sa16_cp8(void* d,const void* s,bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"::"r"(sa16_smem(d)),"l"(s),"r"(ok?8:0));
}
__device__ void sa16_commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ void sa16_wait() { asm volatile("cp.async.wait_group %0;"::"n"(N)); }
__device__ void sa16_ld4(uint32_t* d,const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(sa16_smem(p)));
}
__device__ void sa16_mma(float* c,const uint32_t* a,uint32_t b0,uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// Logical E2M1 converted to BF16 bits by its format definition, not a weight cache/requantizer.
__device__ uint16_t sa16_e2bits(uint32_t c) {
    uint32_t m=c&7,positive=m<2?m*0x3f00u:(((m>>1)+126u)<<7)|((m&1)<<6);
    return (uint16_t)(positive | (m?((c&8)<<12):0)); // canonicalize both encoded zeros, as TS integer decode does
}
// Optional isolated exact-decode ablation. Integer construction preserves the old
// BF16 multiply's signed zero (negative nonzero E2M1 times a zero UE scale).
#ifndef SA16_INTEGER_DECODE
#define SA16_INTEGER_DECODE 0
#endif
__host__ __device__ uint16_t sa16_weight_bits_int(uint32_t code,uint8_t raw) {
    unsigned mag=code&7u,sign=mag?((code&8u)<<12):0u;
    unsigned e=(raw>>3)&15u,m=raw&7u;
    unsigned e2=mag<4?mag:(2u+(mag&1u))<<(mag/2-1);
    unsigned product=(e?8u+m:m)*e2;
    if(!product||raw==127||raw==255)return (uint16_t)sign;
    unsigned h;
#if defined(__CUDA_ARCH__)
    h=31u-(unsigned)__clz(product);
#else
    h=0;for(unsigned q=product;q>1;q>>=1)h++;
#endif
    unsigned adj=e?e+116u:117u;
    return (uint16_t)(sign|((h+adj)<<7)|((product<<(7-h))&127u));
}
__device__ uint32_t sa16_weight_pair(uint32_t lo,uint32_t hi,uint8_t rawScale) {
#if SA16_INTEGER_DECODE
    return (uint32_t)sa16_weight_bits_int(lo,rawScale)|((uint32_t)sa16_weight_bits_int(hi,rawScale)<<16);
#else
    uint32_t bits=(uint32_t)sa16_e2bits(lo)|((uint32_t)sa16_e2bits(hi)<<16);
    __nv_bfloat162 pair=*reinterpret_cast<__nv_bfloat162*>(&bits);
    __nv_bfloat16 scale=__float2bfloat16_rn(2.0f*ts_q35_ue4m3(rawScale));
    pair=__hmul2(pair,__halves2bfloat162(scale,scale));
    return *reinterpret_cast<uint32_t*>(&pair);
#endif
}
extern "C" __global__ void ts_sa16_pack(const float* x,int n,int k,__nv_bfloat16* packed,int wrongPermutation) {
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)16*k;i+=(long long)gridDim.x*blockDim.x) {
        int r=(int)(i/k),l=(int)(i%k),c=wrongPermutation?l:sa16_permute(l);
        packed[i]=__float2bfloat16_rn(r<n?x[(long long)r*k+c]:0.0f);
    }
}
__device__ void sa16_stage(SA16Storage& s,const __nv_bfloat16* x,const uint8_t* w,const uint8_t* scales,int rows,int k,int row0,int tile,int stage) {
    int tid=threadIdx.x,k0=tile*128;
    for(int item=tid;item<256;item+=128) { int r=item/16,c=item%16;sa16_cp16(&s.x[stage][r*136+c*8],x+(long long)r*k+k0+c*8,true); }
    for(int item=tid;item<256;item+=128) { int r=item/4,c=item%4,global=row0+r;sa16_cp16(&s.w[stage][r*80+c*16],w+(long long)(global<rows?global:0)*(k/2)+k0/2+c*16,global<rows); }
    if(tid<64) { int global=row0+tid;sa16_cp8(&s.s[stage][tid*8],scales+(long long)(global<rows?global:0)*(k/16)+k0/16,global<rows); }
}
// A=M16 decoded weight rows, B=N8 activation tokens. N8 uses all 128 C
// elements of one warp MMA; N16 has two token blocks. The raw cp.async stages,
// TS K permutation, split partition, alpha and fixed-order combine are unchanged.
extern "C" __global__ __launch_bounds__(128) void ts_wm16_weightmajor(
    const __nv_bfloat16* x,const uint8_t* w,const uint8_t* scales,int rows,int k,int n,int tilesPerSplit,float alpha,float* y) {
    extern __shared__ __align__(16) uint8_t raw[];SA16Storage& s=*reinterpret_cast<SA16Storage*>(raw);
    int row0=blockIdx.x*64,start=blockIdx.z*tilesPerSplit,end=min(k/128,start+tilesPerSplit),count=end-start;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,q=lane&3;
    float acc[2][4]={{0,0,0,0},{0,0,0,0}};
    for(int z=0;z<3;z++) {if(z<count)sa16_stage(s,x,w,scales,rows,k,row0,start+z,z);sa16_commit();}
    for(int it=0;it<count;it++) {
        sa16_wait<2>();__syncthreads();
        if(it+3<count)sa16_stage(s,x,w,scales,rows,k,row0,start+it+3,(it+3)%4);sa16_commit();
        int st=it%4;
#pragma unroll
        for(int ch=0;ch<4;ch++) {
            int r0=warp*16+g,r1=r0+8;
            uint32_t word0=*(uint32_t*)(&s.w[st][r0*80+ch*16+q*4]);
            uint32_t word1=*(uint32_t*)(&s.w[st][r1*80+ch*16+q*4]);
            uint8_t scale0=s.s[st][r0*8+ch*2+(q>>1)],scale1=s.s[st][r1*8+ch*2+(q>>1)];
#pragma unroll
            for(int step=0;step<2;step++) {
                uint32_t u0=word0>>(16*step),u1=word1>>(16*step);
                uint32_t a[4]={sa16_weight_pair(u0&15,(u0>>4)&15,scale0),sa16_weight_pair(u1&15,(u1>>4)&15,scale1),
                    sa16_weight_pair((u0>>8)&15,(u0>>12)&15,scale0),sa16_weight_pair((u1>>8)&15,(u1>>12)&15,scale1)};
#pragma unroll
                for(int tokenBlock=0;tokenBlock<2;tokenBlock++)if(tokenBlock*8<n) {
                    int token=tokenBlock*8+g,c=ch*32+step*16+q*2;
                    uint32_t b0=*(uint32_t*)(&s.x[st][token*136+c]);
                    uint32_t b1=*(uint32_t*)(&s.x[st][token*136+c+8]);
                    sa16_mma(acc[tokenBlock],a,b0,b1);
                }
            }
        }
    }
    sa16_wait<0>();
#pragma unroll
    for(int tokenBlock=0;tokenBlock<2;tokenBlock++)for(int h=0;h<2;h++) {
        int token=tokenBlock*8+q*2,row=row0+warp*16+g+8*h;
        if(token<n&&row<rows)y[((long long)blockIdx.z*n+token)*rows+row]=acc[tokenBlock][2*h]*(gridDim.z==1?alpha:1.0f);
        if(token+1<n&&row<rows)y[((long long)blockIdx.z*n+token+1)*rows+row]=acc[tokenBlock][2*h+1]*(gridDim.z==1?alpha:1.0f);
    }
}
extern "C" __global__ void ts_sa16_combine(const float* partial,int splits,long long count,float alpha,float* out) {
    for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(long long)gridDim.x*blockDim.x) {float v=0;for(int z=0;z<splits;z++)v+=partial[z*count+i];out[i]=alpha*v;}
}
// Independent logical-index FP64 oracle. Does not use candidate permutation/weight/pack helpers.
extern "C" __global__ void ts_sa16_reference_f64(const float* x,const uint8_t* codes,const uint8_t* scales,int rows,int k,int n,float alpha,double* out,double* absolute) {
    long long output=((long long)blockIdx.x*blockDim.x+threadIdx.x)/32;int lane=threadIdx.x&31;
    if(output>=(long long)n*rows)return;int row=(int)(output%rows),token=(int)(output/rows);double sum=0,ab=0;
    const double table[16]={0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6};
    for(int kk=lane;kk<k;kk+=32) {
        uint8_t byte=codes[(long long)row*(k/2)+(kk/16)*8+(kk&7)],code=(byte>>((kk&8)?4:0))&15,sc=scales[(long long)row*(k/16)+kk/16];
        int e=(sc>>3)&15,m=sc&7;double scale=(sc==127||sc==255)?0:e==0?ldexp((double)m,-9):ldexp(1.0+(double)m/8,e-7);
        uint32_t bits=__float_as_uint(x[(long long)token*k+kk]);bits=(bits+0x7fffu+((bits>>16)&1))&0xffff0000u;
        double term=table[code]*scale*(double)__uint_as_float(bits);sum+=term;ab+=fabs(term);
    }
    for(int off=16;off;off>>=1) {sum+=__shfl_down_sync(0xffffffff,sum,off);ab+=__shfl_down_sync(0xffffffff,ab,off);}
    if(lane==0){out[output]=(double)alpha*sum;absolute[output]=fabs((double)alpha)*ab;}
}
// SA16_EVICTION_SWEEP_BEGIN
// Separate cache-control entry: does not modify candidate/reference arithmetic.
extern "C" __global__ void ts_sa16_cache_sweep(const uint32_t* data,uint64_t words,uint32_t* checksum) {
    const uint64_t id=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    const uint64_t stride=(uint64_t)gridDim.x*blockDim.x;
    uint32_t sum=0;
    for(uint64_t i=id;i<words;i+=stride) {
        uint32_t value;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(data+i) : "memory");
        sum+=value;
    }
    checksum[id]=sum;
}
