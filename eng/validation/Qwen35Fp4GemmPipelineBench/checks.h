#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>
inline void PipelineGuards(const std::vector<uint32_t>& words,std::size_t payload,std::size_t guard,uint32_t marker) {
 if(words.size()!=payload+2*guard)throw std::runtime_error("guarded payload size mismatch");
 for(std::size_t i=0;i<guard;i++)if(words[i]!=marker||words[words.size()-1-i]!=marker)throw std::runtime_error("output guard corruption");
}
inline std::size_t PipelineDifference(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b) {
 if(a.size()!=b.size())throw std::runtime_error("whole-output shape mismatch");
 for(std::size_t i=0;i<a.size();i++)if(a[i]!=b[i])return i;
 return a.size();
}
