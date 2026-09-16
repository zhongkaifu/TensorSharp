#define main fixture_main
#include "dsv41_tp_test.cpp"
#undef main
#include <cuda_runtime.h>
int main() {
    auto init=ggml_init({1024*1024,nullptr,true});ggml_free(init);
    const auto device=ggml_backend_reg_dev_get(ggml_backend_cuda_reg(),0);
    for(auto type:{GGML_TYPE_Q2_K,GGML_TYPE_Q4_K}) {
        fixture full(type,type==GGML_TYPE_Q2_K?GGML_TYPE_Q3_K:GGML_TYPE_Q6_K,5120);
        for(int ranks:{2,7,8}) for(int tokens:{16,65,129}) {
            const auto part=tsg_dsv41_tp::split_weights(full.hidden,full.down.type,ranks,0).back();
            fixture subset(full,part);
            evaluation original(subset,tokens,nullptr,0,device),candidate(subset,tokens,nullptr,0,device,false,false,full.hidden,part.first);
            auto run=[](evaluation& e) {
                const auto start=std::chrono::steady_clock::now();
                e.run();
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            };
            for(int n=0;n<15;++n){run(original);run(candidate);}
            std::vector<double> before,after;
            for(int n=0;n<60;++n){
                if(n%2){after.push_back(run(candidate));before.push_back(run(original));}
                else{before.push_back(run(original));after.push_back(run(candidate));}
            }
            auto arr=[](const std::vector<double>& v){std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';};
            std::cout<<std::setprecision(17)<<"STRIP_BENCH {\"type\":\""<<ggml_type_name(type)<<"\",\"ranks\":"<<ranks<<",\"rows\":"<<part.count<<",\"tokens\":"<<tokens<<",\"before_ms\":";
            arr(before);std::cout<<",\"after_ms\":";arr(after);std::cout<<'}'<<std::endl;
        }
    }
}
