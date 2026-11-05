// Standalone exact F16 intermediate only; no quantizer or production integration.
// The generated plain helper is a verbatim ts_q35_tile_to_h except its name
// and global destination row stride. Its i8-to-half/hmul2 operations are original.
extern "C" __global__ void ts_bench_decode_plain(
    uint8_t* const* table, int maxBlocks, long long kOff, long long vOff,
    const int* keyEnd, int rowsPerSeq, int keys, int nKv, half* out)
{
    const int seq = blockIdx.z >> 1, isV = blockIdx.z & 1, head = blockIdx.y;
    const int first = blockIdx.x * 32, key = first + (threadIdx.x & 31);
    const int validKeys = keyEnd[(long long)seq * rowsPerSeq + rowsPerSeq - 1];
    if (key >= validKeys) return;
    const uint8_t* raw = table[(long long)seq * maxBlocks + first / TS_Q35_KV_BLOCK]
        + ((long long)head * TS_Q35_KV_BLOCK + first % TS_Q35_KV_BLOCK) * TS_Q35_Q8_ROW
        + (isV ? vOff : kOff);
    half* dst = out + (((long long)seq * 2 + isV) * nKv + head) * keys * TS_Q35_HD
        + (long long)first * TS_Q35_HD;
    ts_bench_tile_to_plain(raw, dst, threadIdx.x);
}

// Strict-only independent reference uses the untouched legacy helper and its
// original264-half destination stride. The padding is never written.
extern "C" __global__ void ts_bench_decode_reference(
    uint8_t* const* table, int maxBlocks, long long kOff, long long vOff,
    const int* keyEnd, int rowsPerSeq, int keys, int nKv, half* out)
{
    const int seq = blockIdx.z >> 1, isV = blockIdx.z & 1, head = blockIdx.y;
    const int first = blockIdx.x * 32, key = first + (threadIdx.x & 31);
    const int validKeys = keyEnd[(long long)seq * rowsPerSeq + rowsPerSeq - 1];
    if (key >= validKeys) return;
    const uint8_t* raw = table[(long long)seq * maxBlocks + first / TS_Q35_KV_BLOCK]
        + ((long long)head * TS_Q35_KV_BLOCK + first % TS_Q35_KV_BLOCK) * TS_Q35_Q8_ROW
        + (isV ? vOff : kOff);
    half* dst = out + (((long long)seq * 2 + isV) * nKv + head) * keys * TS_Q35_ATTN_ROW
        + (long long)first * TS_Q35_ATTN_ROW;
    ts_q35_tile_to_h(raw, dst, threadIdx.x);
}

struct TsBenchHalfBanks
{
    half kv[2][2][TS_Q35_AI_KT * TS_Q35_ATTN_ROW];
};
static_assert(sizeof(TsBenchHalfBanks) == 67584, "double half-bank layout changed");
