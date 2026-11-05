// CPU-only independent logical-nibble reference, no CUDA intrinsics or reinterpret casts.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
static std::vector<uint8_t> PackNibbleReference(const std::vector<uint8_t>& source) {
    if(source.size()%4)throw std::runtime_error("word-aligned codes required");
    std::vector<uint8_t> result(source.size(),0);
    for(size_t at=0;at<source.size();at+=4)
        for(int byte=0;byte<4;byte++) {
            result[at+byte/2] |= uint8_t((source[at+byte]&15)<<(4*(byte%2)));
            result[at+2+byte/2] |= uint8_t((source[at+byte]>>4)<<(4*(byte%2)));
        }
    return result;
}
static std::vector<uint8_t> UnpackNibbleReference(const std::vector<uint8_t>& packed) {
    if(packed.size()%4)throw std::runtime_error("word-aligned packed codes required");
    std::vector<uint8_t> result(packed.size(),0);
    for(size_t at=0;at<packed.size();at+=4)
        for(int byte=0;byte<4;byte++)
            result[at+byte] = uint8_t(((packed[at+byte/2]>>(4*(byte%2)))&15)|(((packed[at+2+byte/2]>>(4*(byte%2)))&15)<<4));
    return result;
}
