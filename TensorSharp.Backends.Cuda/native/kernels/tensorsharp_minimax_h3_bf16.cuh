// TensorSharp-owned BF16 storage conversion for H3's dense projections.
// Included last so existing PTX entry bodies retain their compiler label IDs.
extern "C" __global__ void ts_h3_weight_bf16_f32(const unsigned short * x,float * y,int count) {
    TS_PDL_ENTRY();
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)y[i]=__uint_as_float(static_cast<unsigned int>(x[i])<<16);
}
