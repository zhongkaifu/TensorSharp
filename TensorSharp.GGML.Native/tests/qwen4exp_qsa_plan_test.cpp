// Independent single-holder QSA layout checks. No ggml, tensor payload or GPU.
// Contract reviewed against llama-memory-hybrid-idx.cpp:set_input_qsa at
// d1d3c3396aa13a5f239109a822666c4870490ad5. Nonzero position offsets are an
// explicit TensorSharp specialization; foreign sequences/cache holes are absent.
#include "ggml_ops_qwen4exp_qsa.h"
#include <cmath>
#include <iostream>
#include <string>

static int checks = 0;
static void require(bool value, const std::string& name)
{
    ++checks;
    if (!value) throw std::runtime_error(name);
}
template<class T> static void equal(const std::vector<T>& actual, const std::vector<T>& expected, const char* name)
{ require(actual == expected, name); }
static std::vector<int32_t> text_positions(int count, int offset = 0)
{
    std::vector<int32_t> result;
    for (int i = 0; i < count; ++i)
        for (int axis = 0; axis < 3; ++axis) result.push_back(offset + i);
    return result;
}
template<class F> static void invalid(F action, const char* name)
{
    bool caught = false;
    try { action(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, name);
}

static void contiguous_residues()
{
    // Every remainder, including no incomplete tail and ratio64's final bit.
    for (int ratio : {1, 2, 4, 64})
        for (int remainder = 0; remainder < ratio; ++remainder)
            for (int extra_padding : {0, 3})
            {
                const int live = 2 * ratio + remainder, padded = live + extra_padding;
                auto positions = text_positions(live);
                const auto before = positions;
                auto plan = q4e_qsa_plan(positions.data(), live, padded, 0, live, ratio);
                const int blocks = (padded + ratio - 1) / ratio, complete = live / ratio;
                require(!plan.ranked, "ordinary text must retain absolute positions");
                equal(positions, before, "input history is immutable");
                require(plan.cell_blocks.size() == (size_t)padded, "cell map has padded extent");
                require(plan.block_cells.size() == (size_t)blocks * ratio, "pool members have block extent");
                for (int cell = 0; cell < padded; ++cell)
                {
                    const int expected = cell < complete * ratio ? cell / ratio : complete;
                    require(plan.cell_blocks[cell] == expected, "complete cells map by ratio; all others map to spare");
                }
                for (int block = 0; block < blocks; ++block)
                {
                    for (int member = 0; member < ratio; ++member)
                        require(plan.block_cells[block * ratio + member] ==
                            (block < complete ? block * ratio + member : 0), "complete member order / safe spare gather");
                    for (int axis = 0; axis < 4; ++axis)
                        require(plan.block_positions[axis * blocks + block] ==
                            (block < complete ? block * ratio : 0), "block rotary uses first absolute position");
                }
                for (int query = 0; query < live; ++query)
                {
                    for (int axis = 0; axis < 4; ++axis)
                        require(plan.query_positions[axis * live + query] == query, "query planar THWT order");
                    for (int block = 0; block < blocks; ++block)
                    {
                        float actual = plan.bias[query * blocks + block];
                        if (block < complete)
                        {
                            // A completed prefix contains floor((q+1)/ratio)
                            // blocks. Everything else is the protected tail;
                            // the attention mask, not this bias, removes future cells.
                            float expected = block < (query + 1) / ratio ? 0.f : 1e9f;
                            require(actual == expected, "query-dependent completed-prefix/tail bias");
                        }
                        else if (block == complete)
                            require(actual == 1e9f, "unpooled tail remains finite even before one complete block");
                        else require(std::isinf(actual) && actual < 0, "unused block remains negative infinity");
                        require(!std::isnan(actual), "bias contains no NaN");
                    }
                }
            }
}

static void explicit_layouts()
{
    auto short_text = text_positions(3);
    auto short_plan = q4e_qsa_plan(short_text.data(), 3, 8, 1, 2, 4);
    equal(short_plan.cell_blocks, std::vector<int32_t>(8, 0), "short prefix uses only spare block");
    equal(short_plan.block_cells, std::vector<int32_t>(8, 0), "short prefix never pools missing keys");
    require(short_plan.bias[0] == 1e9f && short_plan.bias[2] == 1e9f,
        "short prefix query rows both have finite tail");
    equal(short_plan.query_positions, std::vector<int32_t>{1,2,1,2,1,2,1,2}, "query offset/planar layout");

    // Absolute positions101..111: the first three are incomplete, followed by
    // complete blocks104..107 and108..111. This is intentionally not rebased.
    auto offset = text_positions(11, 101);
    auto p = q4e_qsa_plan(offset.data(), 11, 16, 2, 1, 4);
    equal(p.cell_blocks, std::vector<int32_t>{2,2,2,0,0,0,0,1,1,1,1,2,2,2,2,2}, "nonzero offset compact bucket map");
    equal(p.block_cells, std::vector<int32_t>{3,4,5,6,7,8,9,10,0,0,0,0,0,0,0,0}, "offset complete members");
    equal(p.block_positions, std::vector<int32_t>{104,108,0,0,104,108,0,0,104,108,0,0,104,108,0,0}, "offset first-member rotary");
    require(p.bias[0] == 1e9f && p.bias[1] == 1e9f && p.bias[2] == 1e9f && std::isinf(p.bias[3]),
        "offset query103 cannot mark blocks104/108 as completed prefix");

    // Arrival order differs from THW order. Equal coordinates at cells3/6
    // retain arrival order; both queries share the same upper-bound tail.
    std::vector<int32_t> media{0,0,0, 4,1,0, 4,0,1, 4,0,0, 4,1,1, 5,5,5, 4,0,0};
    auto m = q4e_qsa_plan(media.data(), 7, 8, 0, 7, 2);
    require(m.ranked, "repeated media time uses THW ranking");
    equal(m.block_cells, std::vector<int32_t>{0,3,6,2,1,4,0,0}, "THW rank pool membership");
    equal(m.cell_blocks, std::vector<int32_t>{0,2,1,0,2,3,1,3}, "ranked cells map back to physical cache");
    equal(m.block_positions, std::vector<int32_t>{0,4,4,0, 0,0,1,0, 0,0,0,0, 0,4,4,0}, "media block rotary preserves THWT");
    for (int q : {3,6})
        equal(std::vector<float>(m.bias.begin()+q*4,m.bias.begin()+(q+1)*4),
            std::vector<float>{0,1e9f,1e9f,1e9f}, "equal THW queries use identical causal tail");
    require(m.query_positions[1*7+1] == 1 && m.query_positions[2*7+2] == 1,
        "query height and width are distinct axes");
}

static void invalid_inputs()
{
    auto p = text_positions(4);
    invalid([&]{ q4e_qsa_plan(nullptr,4,4,0,1,4); }, "null history");
    invalid([&]{ q4e_qsa_plan(p.data(),0,4,0,1,4); }, "empty history");
    invalid([&]{ q4e_qsa_plan(p.data(),4,3,0,1,4); }, "padding smaller than live");
    invalid([&]{ q4e_qsa_plan(p.data(),4,4,-1,1,4); }, "negative query start");
    invalid([&]{ q4e_qsa_plan(p.data(),4,4,0,0,4); }, "empty query");
    invalid([&]{ q4e_qsa_plan(p.data(),4,4,3,2,4); }, "query exceeds live range");
    invalid([&]{ q4e_qsa_plan(p.data(),4,4,0,1,0); }, "zero ratio");
    invalid([&]{ q4e_qsa_plan(p.data(),4,4,0,1,65); }, "ratio exceeds supported bound");
    invalid([&]{ q4e_qsa_plan(p.data(),4,INT32_MAX,0,1,4); }, "rounded padded extent overflow before allocation");
    invalid([&]{ q4e_qsa_plan(p.data(),4,INT32_MAX-1,0,1,1); }, "planar position extent overflow before allocation");
    for (int axis = 0; axis < 3; ++axis)
    {
        auto bad = p; bad[axis] = -1;
        invalid([&]{ q4e_qsa_plan(bad.data(),4,4,0,1,4); }, "negative THW coordinate");
    }
    auto first = q4e_qsa_plan(p.data(),4,4,0,4,4);
    p[0] = 3;
    auto second_positions = text_positions(5, 100);
    auto second = q4e_qsa_plan(second_positions.data(),5,8,4,1,4);
    require(first.query_positions[0] == 0 && first.cell_blocks.size() == 4,
        "plans own their arrays across source mutation and another holder plan");
    require(second.query_positions[0] == 104 && second.cell_blocks.size() == 8,
        "independent holder history has independent coordinates and padding");
}

int main()
{
    try
    {
        contiguous_residues(); explicit_layouts(); invalid_inputs();
        std::cout << "PASS " << checks << " independent QSA plan checks; no model/native/GPU execution\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n';
        return 1;
    }
}
