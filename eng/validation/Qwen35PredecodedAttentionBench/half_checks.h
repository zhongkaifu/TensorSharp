// Host-only strict intermediate checker shared by the CUDA runner and CPU tests.
#pragma once
#include <cstdint>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

struct DecodedHalfMismatch { std::size_t row; int column; };
inline std::optional<DecodedHalfMismatch> FindDecodedHalfMismatch(
    const std::vector<uint16_t>& expected,const std::vector<uint16_t>& actual,
    int sequences,int heads,int keys,const std::vector<int>& validEnds,int actualStride,
    uint16_t expectedPoison,uint16_t actualPoison)
{
    if(sequences<=0||heads<=0||keys<=0||(actualStride!=256&&actualStride!=264)
        ||validEnds.size()!=std::size_t(sequences))throw std::invalid_argument("invalid half layout");
    std::size_t rows=std::size_t(sequences)*2*heads*keys;
    if(expected.size()!=rows*264||actual.size()!=rows*actualStride)throw std::invalid_argument("half buffer size differs");
    for(int end:validEnds)if(end<0||end>keys)throw std::invalid_argument("invalid sequence key bound");
    for(int seq=0;seq<sequences;seq++)for(int plane=0;plane<2;plane++)for(int h=0;h<heads;h++)for(int key=0;key<keys;key++) {
        std::size_t row=(((std::size_t)seq*2+plane)*heads+h)*keys+key;
        bool valid=key<validEnds[seq];
        for(int d=0;d<264;d++) {
            uint16_t wanted=expected[row*264+d];
            bool fail=(d>=256||!valid)&&wanted!=expectedPoison;
            if(d<actualStride) {
                uint16_t got=actual[row*actualStride+d];
                fail|=(valid&&d<256)?got!=wanted:got!=actualPoison;
            }
            if(fail)return DecodedHalfMismatch{row,d};
        }
    }
    return std::nullopt;
}
