// Optional all-data/L2 eviction control, outside the measured complete quantizer+GEMV interval.
// This is not a guarantee of physical cold cache or a weights-only experiment.
#include <memory>
struct CacheSweep {
    static constexpr unsigned long long Value=0x5a5a5a5aull;
    CUfunction function=nullptr;
    std::unique_ptr<Buffer> data,checks;
    size_t bytes=0;unsigned long long words=0,calls=0;int l2=0,blocks=0;
    explicit CacheSweep(Module& module,int sms) {
        CUdevice device;Cu(cuDeviceGet(&device,0));
        Cu(cuDeviceGetAttribute(&l2,CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,device));
        if(l2<=0||sms<=0||size_t(l2)>(SIZE_MAX-255)/2)throw std::runtime_error("invalid actual driver L2 capacity");
        bytes=(size_t(l2)*2+255)/256*256;words=bytes/sizeof(uint32_t);
        blocks=int(std::min<size_t>(size_t(sms)*4,(size_t(words)+255)/256));
        if(blocks<1||words>UINT64_MAX/Value)throw std::runtime_error("cache sweep arithmetic overflow");
        data=std::make_unique<Buffer>(bytes);checks=std::make_unique<Buffer>(size_t(blocks)*2*sizeof(unsigned long long));
        Rt(cudaMemset(data->Data(),0x5a,bytes));
        Cu(cuModuleGetFunction(&function,module.module,"ts_bench_q6_cache_sweep"));
        int maxThreads=0,active=0,regs=0,local=0,shared=0;
        Cu(cuFuncGetAttribute(&maxThreads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,function));
        Cu(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
        Cu(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
        Cu(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
        Cu(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,256,0));
        if(maxThreads<256||active<1||regs<=0||regs>255||local!=0)throw std::runtime_error("cache sweep entry not launchable without local spills");
        std::printf("{\"event\":\"cache_sweep_setup\",\"mode\":\"all_data_l2_eviction_control\",\"driver_l2_bytes\":%d,\"sweep_bytes\":%llu,\"words\":%llu,\"blocks\":%d,\"block_threads\":256,\"method\":\"volatile_asm_ld_global_cg_u32_grid_stride_per_block_u64_sum_and_count\",\"physical_cold_guaranteed\":false,\"excluded_from_event_interval\":true,\"registers\":%d,\"local_bytes\":%d,\"static_shared_bytes\":%d,\"active_ctas\":%d,\"max_block_threads\":%d}\n",l2,(unsigned long long)bytes,words,blocks,regs,local,shared,active,maxThreads);
    }
    void Before(const Shape& shape,const Variant& measured,const char* candidate,int round,const char* half) {
        if(!candidate||!half||round<0)throw std::runtime_error("missing cache sweep half identity");
        Rt(cudaMemset(checks->Data(),0xa5,checks->bytes));
        void* input=data->Data();void* output=checks->Data();void* args[]={&input,&words,&output};
        Cu(cuLaunchKernel(function,blocks,1,1,256,1,1,0,nullptr,args,nullptr));
        Rt(cudaDeviceSynchronize());
        std::vector<unsigned long long> got(size_t(blocks)*2);
        Rt(cudaMemcpy(got.data(),checks->Data(),checks->bytes,cudaMemcpyDeviceToHost));
        unsigned long long stride=(unsigned long long)blocks*256,full=words/stride,remainder=words%stride,total=0,sum=0;
        for(int b=0;b<blocks;b++) {
            unsigned long long begin=(unsigned long long)b*256;
            unsigned long long expected=full*256+(remainder>begin?std::min<unsigned long long>(256,remainder-begin):0);
            if(got[2*b]!=Value*expected||got[2*b+1]!=expected)throw std::runtime_error("cache sweep full-read checksum/count mismatch");
            sum+=got[2*b];total+=got[2*b+1];
        }
        if(total!=words||sum!=Value*words)throw std::runtime_error("cache sweep coverage mismatch");
        data->CheckGuard();checks->CheckGuard();++calls;
        std::printf("{\"event\":\"cache_sweep\",\"index\":%llu,\"candidate\":%s,\"measured_symbol\":%s,\"shape\":%s,\"plane_role\":%s,\"round\":%d,\"half\":%s,\"driver_l2_bytes\":%d,\"sweep_bytes\":%llu,\"words_read\":%llu,\"checksum\":%llu,\"guards\":true,\"checksum_verified\":true,\"synchronized_before_event\":true,\"excluded_from_event_interval\":true}\n",calls,Quote(candidate).c_str(),Quote(measured.symbol).c_str(),Quote(Name(shape)).c_str(),Quote(TimingPlane).c_str(),round,Quote(half).c_str(),l2,(unsigned long long)bytes,total,sum);
    }
};
