// TensorSharp-owned F32 MiniMax-H3 VAE kernels. Upstream ggml is unchanged.
#include <cuda_fp16.h>

extern "C" __global__ void ts_h3_weight_f16_f32(const half * x,float * y,int count) {
    TS_PDL_ENTRY();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=__half2float(x[i]);
}

extern "C" __global__ void ts_h3_copy_strided_f32(const float * x, float * y, int count,
        long long n0, long long n1, long long n2, long long n3,
        long long xs0, long long xs1, long long xs2, long long xs3,
        long long ys0, long long ys1, long long ys2, long long ys3) {
    TS_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    long long r=i, c3=r%n3; r/=n3; long long c2=r%n2; r/=n2;
    long long c1=r%n1, c0=r/n1;
    y[c0*ys0+c1*ys1+c2*ys2+c3*ys3] = x[c0*xs0+c1*xs1+c2*xs2+c3*xs3];
}

extern "C" __global__ void ts_h3_clamp_f32(const float * x, float * y, int count, float lower, float upper) {
    TS_PDL_ENTRY();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) { const float v=x[i]; y[i]=v<lower ? lower : (v>upper ? upper : v); }
}

extern "C" __global__ void ts_h3_transform_f32(const float * x, const float * a,
        const float * b, float * y, int count, int op, int p0, int p1, int p2,
        int p3, int p4, int p5, int p6, int p7, int p8, int p9) {
    TS_PDL_ENTRY();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (op == 0) { // reflect im2col: x[H,W,C], y[bandRows*OW,C*K*K]
        int dim = p0*p3*p3, row = i/dim, tap = i%dim;
        int channel = tap/(p3*p3), ky = (tap/p3)%p3, kx = tap%p3;
        int sy = (p8 + row/p4)*p7 - p6 + ky, sx = (row%p4)*p7 - p5 + kx;
        sy = p1 <= 1 ? 0 : (sy < 0 ? -sy : (sy >= p1 ? 2*(p1-1)-sy : sy));
        sx = p2 <= 1 ? 0 : (sx < 0 ? -sx : (sx >= p2 ? 2*(p2-1)-sx : sx));
        y[i] = x[(sy*p2+sx)*p0+channel];
    } else if (op == 1) { // audio im2col: x[T,C], y[band,C*K]
        int dim = p0*p2, row = i/dim+p6, tap = i%dim, channel = tap/p2, k = tap%p2;
        int st = p7 ? row-k : row*p3-p4+k*p5;
        bool valid = !p7 || st%p3 == 0;
        if (p7) st /= p3;
        y[i] = valid && st >= 0 && st < p1 ? x[st*p0+channel] : 0.0f;
    } else if (op == 2) { // audio shared filter, optional zero-stuff + replicate
        int ch = i%p0, ot = i/p0;
        float sum = 0.0f;
        for (int k = 0; k < p2; ++k) {
            int st = ot*p3+p7-p4+k;
            if (p5) { // upsample: pad source before zero-stuff, zero beyond it
                if (st < 0 || st >= (p1+2*p6)*2 || st%2) continue;
                st = st/2-p6;
            }
            st = max(0, min(p1-1, st));
            sum += a[k]*x[st*p0+ch];
        }
        y[i] = sum;
    } else if (op == 3) { // Snake / SnakeBeta; b is the precomputed reciprocal
        int ch = i%p0;
        float value = x[i], s = sinf(a[ch]*value);
        y[i] = value + (s*s)*b[ch];
    } else if (op == 4) { // mean over heads, then adaptive mean over head dim
        int row = i/p2, channel = i%p2, group = p1/p2;
        float pooled = 0.0f;
        for (int j = 0; j < group; ++j) {
            float heads = 0.0f;
            for (int h = 0; h < p0; ++h) heads += x[row*p0*p1+h*p1+channel*group+j];
            pooled += heads/p0;
        }
        y[i] = pooled/group;
    } else if (op == 5) { // exact-form GELU, matching the managed A&S polynomial
        const float value=x[i], scaled=value*0.7071067811865475f;
        const float absolute=fabsf(scaled), t=1.0f/(1.0f+0.3275911f*absolute);
        const float polynomial=((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;
        const float sign=scaled > 0.0f ? 1.0f : (scaled < 0.0f ? -1.0f : 0.0f);
        const float erf=sign*(1.0f-polynomial*expf(-absolute*absolute));
        y[i]=value*(0.5f*(1.0f+erf));
    }
}

extern "C" __global__ void ts_h3_groupnorm_silu_f32(const float * x, const float * gain,
        const float * bias, float * y, int rows, int channels, int groups, float eps) {
    TS_PDL_ENTRY();
    int frame = blockIdx.x/groups, group = blockIdx.x%groups, width = channels/groups;
    int n = rows*width;
    __shared__ double sums[256];
    double sum = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += x[(frame*rows+i/width)*channels+group*width+i%width];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sums[threadIdx.x] += sums[threadIdx.x+s];
        __syncthreads();
    }
    float mean = float(sums[0]/n);
    sum = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[(frame*rows+i/width)*channels+group*width+i%width]-mean;
        sum += v*v;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sums[threadIdx.x] += sums[threadIdx.x+s];
        __syncthreads();
    }
    float scale = 1.0f/sqrtf(float(sums[0]/n)+eps);
    // Input and output can alias. All reads contributing to statistics finish first.
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int ch = group*width+i%width, offset = (frame*rows+i/width)*channels+ch;
        float v = (x[offset]-mean)*scale;
        if (gain) v *= gain[ch];
        if (bias) v += bias[ch];
        y[offset] = v/(1.0f+expf(-v));
    }
}
