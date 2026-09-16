#include "ggml_ops_matmul_precision.cpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
int main() {
    ggml_init_params params{1024*1024,nullptr,false};
    auto* ctx=ggml_init(params);
    const auto dot=ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
    volatile float sink=0;
    for(int n:{128,256,1024,2048,4096,5120,8192}) {
      std::vector<float>a(n*64),b(n*64);
      std::mt19937 r(4321);std::uniform_real_distribution<float>u(-1,1);
      for(auto&x:a)x=u(r);for(auto&x:b)x=u(r);
      std::vector<double>old_ms,new_ms;
      for(int pair=0;pair<12;++pair) for(int order=0;order<2;++order) {
        bool owned=(order+pair)%2;float sum=0;
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<20000;++i){int row=i%64;float value;
          if(owned)value=paired_f32_dot(a.data()+row*n,b.data()+row*n,n);
          else dot(n,&value,0,a.data()+row*n,0,b.data()+row*n,0,1);
          sum+=value;
        }
        sink=sum;
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        (owned?new_ms:old_ms).push_back(ms);
      }
      auto median=[](std::vector<double>x){std::sort(x.begin(),x.end());return(x[5]+x[6])/2;};
      printf("n=%d old_ms=%.4f new_ms=%.4f ratio=%.4f\n",n,median(old_ms),median(new_ms),median(new_ms)/median(old_ms));
      for(int p=0;p<12;++p)printf("pair=%d old_ms=%.6f new_ms=%.6f\n",p,old_ms[p],new_ms[p]);
    }
    ggml_free(ctx);
}
