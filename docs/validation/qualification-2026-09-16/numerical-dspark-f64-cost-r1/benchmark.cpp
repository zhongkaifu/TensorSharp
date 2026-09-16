#include "ggml_ops_matmul_precision.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "precision_test_utils.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

struct tsg_matmul_cuda_state_candidate;
tsg_matmul_cuda_state_candidate * tsg_matmul_cuda_init_candidate(ggml_backend_t);
void tsg_matmul_cuda_free_candidate(tsg_matmul_cuda_state_candidate *);
void tsg_matmul_cuda_compute_candidate(tsg_matmul_cuda_state_candidate *, ggml_tensor *);

int main() {
    auto * backend = ggml_backend_cuda_init(0);
    require(backend, "No CUDA backend");
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    auto stream = context->stream(context->device, 0);
    auto * original = tsg_matmul_cuda_init(backend);
    auto * candidate = tsg_matmul_cuda_init_candidate(backend);
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
    struct shape { bool indexed; int inner, rows, tokens, used; ggml_type type; };
    const shape cases[] = {
        {false,256,256,5,1,GGML_TYPE_F32},
        {false,4096,512,1,1,GGML_TYPE_F32}, {false,4096,512,5,1,GGML_TYPE_F32},
        {false,4096,512,8,1,GGML_TYPE_F32}, {false,5120,512,5,1,GGML_TYPE_F32},
        {false,4096,512,5,1,GGML_TYPE_F16},
        {true,256,256,5,2,GGML_TYPE_F32},
        {true,5120,512,1,6,GGML_TYPE_F32}, {true,5120,512,5,6,GGML_TYPE_F32},
        {true,5120,512,17,6,GGML_TYPE_F32},
        {true,5120,512,5,6,GGML_TYPE_F16}, {true,5120,512,5,6,GGML_TYPE_BF16}
    };
    for (const auto & c : cases) {
        auto * ctx = ggml_init({4*1024*1024,nullptr,true});
        input_tensor a(ctx,c.type,{c.inner,c.rows,c.indexed?8:1,1});
        input_tensor b(ctx,GGML_TYPE_F32,{c.inner,c.indexed?1:c.tokens,c.indexed?c.tokens:1,1});
        input_tensor ids(ctx,GGML_TYPE_I32,{c.indexed?c.used:1,c.indexed?c.tokens:1,1,1});
        auto * output = c.indexed ? tsg_matmul_id_f32(ctx,a.tensor,b.tensor,ids.tensor)
                                 : tsg_matmul_f32(ctx,a.tensor,b.tensor);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
        require(buffer,"Cannot allocate benchmark");
        std::mt19937 random(49383);
        std::uniform_real_distribution<float> uniform(-1,1);
        for (auto & value : a.values) value=uniform(random);
        for (auto & value : b.values) value=uniform(random);
        for(int t=0;t<ids.tensor->ne[1];++t)
            for(int e=0;e<ids.tensor->ne[0];++e) ids.values[ids.logical(e,t,0,0)]=(t+e)%8;
        a.upload();b.upload();ids.upload();
        auto run=[&](bool after) {
            if(after)tsg_matmul_cuda_compute_candidate(candidate,output);
            else tsg_matmul_cuda_compute(original,output);
        };
        std::vector<float> before(ggml_nelements(output)),after(before.size());
        run(false);CUDA_CHECK(cudaStreamSynchronize(stream));
        ggml_backend_tensor_get(output,before.data(),0,before.size()*sizeof(float));
        run(true);CUDA_CHECK(cudaStreamSynchronize(stream));
        ggml_backend_tensor_get(output,after.data(),0,after.size()*sizeof(float));
        double old_error=0,new_error=0,scale=0;
        for(int t=0;t<c.tokens;++t) for(int e=0;e<c.used;++e) for(int row=0;row<c.rows;row+=17) {
            double reference=0;
            for(int k=0;k<c.inner;++k)
                reference+=double(a.at(k,row,c.indexed?(t+e)%8:0))*b.at(k,c.indexed?0:t,c.indexed?t:0);
            const int index=row+c.rows*(e+c.used*t);
            old_error+=std::pow(double(before[index])-reference,2);
            new_error+=std::pow(double(after[index])-reference,2);scale+=reference*reference;
        }
        for(int i=0;i<10;++i){run(false);run(true);}
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> times[2];
        for(int pair=0;pair<40;++pair)for(int turn=0;turn<2;++turn) {
            const bool after=(pair+turn)%2;
            CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int repeat=0;repeat<20;++repeat)run(after);
            CUDA_CHECK(cudaEventRecord(end,stream));CUDA_CHECK(cudaEventSynchronize(end));
            float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[after].push_back(ms/20);
        }
        std::cout<<std::setprecision(17)<<"SUM_BENCH {\"indexed\":"<<c.indexed<<",\"inner\":"<<c.inner
            <<",\"rows\":"<<c.rows<<",\"tokens\":"<<c.tokens<<",\"used\":"<<c.used<<",\"type\":\""<<ggml_type_name(c.type)
            <<"\",\"before_relative_l2\":"<<std::sqrt(old_error/scale)<<",\"after_relative_l2\":"<<std::sqrt(new_error/scale);
        for(int type=0;type<2;++type) {
            std::cout<<(type?",\"after_ms\":[":",\"before_ms\":[");
            for(size_t i=0;i<times[type].size();++i){if(i)std::cout<<',';std::cout<<times[type][i];}
            std::cout<<']';
        }
        std::cout<<"}"<<std::endl;
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
    tsg_matmul_cuda_free_candidate(candidate);tsg_matmul_cuda_free(original);ggml_backend_free(backend);
}
