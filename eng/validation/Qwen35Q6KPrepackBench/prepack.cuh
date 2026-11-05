// Independent integer-only preprocessing. Original half d and all signed scales are copied as bytes.
extern "C" __global__ void ts_bench_q6_prepack(const uint8_t* raw, uint8_t* packed, int rows, int k) {
    const unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
#ifdef TS_Q6_BYTE_PLANES
    const unsigned long long blocks=(unsigned long long)rows*(k/256),bytes=blocks*210;
    if(i>=bytes)return;
    const unsigned long long block=i/210;const int at=i%210;
    const unsigned long long dst=at<128?block*128+at:at<192?blocks*128+block*64+at-128:at<208?blocks*192+block*16+at-192:blocks*208+block*2+at-208;
    packed[dst]=raw[i];
#elif defined(TS_Q6_ALIGNED_BLOCKS)
    const unsigned long long blocks=(unsigned long long)rows*(k/256),bytes=blocks*212;
    if(i>=bytes)return;
    const unsigned long long block=i/212;const int at=i%212;
    packed[i]=at<210?raw[block*210+at]:0;
#else
    const unsigned long long count=(unsigned long long)rows*k;
    if(i>=count)return;
    const unsigned long long r=i/k; const int col=i%k, sb=col/256, j=col%256;
    const uint8_t* b=raw+r*(k/256)*210+(unsigned long long)sb*210;
    const int h=j/128,m=(j%128)/32,l=j%32;
    const unsigned int lo=(b[h*64+(m&1)*32+l]>>(m>=2?4:0))&15;
    const unsigned int hi=(b[128+h*32+l]>>(2*m))&3;
    packed[i]=(uint8_t)(int(lo|(hi<<4))-32);
    if(j%16==0)packed[count+r*(k/16)+col/16]=b[192+j/16];
    if(j==0) {
        const unsigned long long at=count+(unsigned long long)rows*(k/16)+(r*(k/256)+sb)*2;
        packed[at]=b[208];packed[at+1]=b[209];
    }
#endif
}
// Optional all-data eviction control, no guarantee of physical cold cache.
extern "C" __global__ void ts_bench_q6_cache_sweep(const uint32_t* input,unsigned long long words,unsigned long long* out) {
    unsigned long long sum=0,count=0;
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=(unsigned long long)gridDim.x*blockDim.x) {
        uint32_t value;asm volatile("ld.global.cg.u32 %0, [%1];":"=r"(value):"l"(input+i):"memory");sum+=value;count++;
    }
    __shared__ unsigned long long sums[256],counts[256];sums[threadIdx.x]=sum;counts[threadIdx.x]=count;__syncthreads();
    for(int d=128;d;d>>=1){if(threadIdx.x<d){sums[threadIdx.x]+=sums[threadIdx.x+d];counts[threadIdx.x]+=counts[threadIdx.x+d];}__syncthreads();}
    if(threadIdx.x==0){out[2*blockIdx.x]=sums[0];out[2*blockIdx.x+1]=counts[0];}
}
