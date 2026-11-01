// Host-only strict buffer coverage checks, shared by the CUDA runner and CPU tests.
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

inline void CheckGuardedWords(const std::vector<uint32_t>& words, std::size_t count, std::size_t guard,
    uint32_t sentinel, bool mustWrite, const char* name)
{
    if(words.size()!=count+2*guard)throw std::runtime_error(std::string(name)+" buffer shape mismatch");
    for(std::size_t i=0;i<guard;i++)
        if(words[i]!=sentinel||words[words.size()-1-i]!=sentinel)
            throw std::runtime_error(std::string(name)+" guard corruption");
    for(std::size_t i=0;i<count;i++)
        if(mustWrite ? words[guard+i]==sentinel : words[guard+i]!=sentinel)
            throw std::runtime_error(std::string(name)+(mustWrite?" unwritten element ":" unexpected write ")+std::to_string(i));
}
