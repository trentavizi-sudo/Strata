// kv_grow_plan_test: the elastic K/V's slot accounting (include/strata/core/kv_grow_plan.hpp), host only.
#include "strata/core/kv_grow_plan.hpp"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::core::kvgrow;
    // round_cells
    CHECK(round_cells(0, 8192, 131072) == 0);
    CHECK(round_cells(1, 8192, 131072) == 8192);
    CHECK(round_cells(8192, 8192, 131072) == 8192);
    CHECK(round_cells(8193, 8192, 131072) == 16384);
    CHECK(round_cells(200000, 8192, 131072) == 131072);
    CHECK(round_cells(5, 0, 131072) == 5);   // a step of 0 counts as 1
    // target_cells: the furthest row, the rows the window appends, the margin
    const int64_t pos[4] = {100, 9000, 40, 8100};
    CHECK(target_cells(pos, 4, 1, 64, 8192, 131072) == 16384);        // 9000 + 65 -> 16384
    CHECK(target_cells(pos, 4, 1, 64, 1, 131072) == 9065);
    CHECK(target_cells(pos, 3, 1, 64, 8192, 131072) == 16384);        // the furthest is still row 1
    CHECK(target_cells(pos, 1, 1, 64, 8192, 131072) == 8192);
    CHECK(target_cells(pos, 0, 1, 64, 8192, 131072) == 0);
    CHECK(target_cells(nullptr, 4, 1, 64, 8192, 131072) == 0);
    CHECK(target_cells(pos, 4, 1, 64, 8192, 8192) == 8192);           // never past the context
    // slots_hold
    const int64_t held[3] = {0, 5000, 300};
    CHECK(slots_hold(held, 3, 256) == 5256);
    CHECK(slots_hold(held, 1, 256) == 0);
    CHECK(slots_hold(held, 0, 256) == 0);
    // chunk_range: whole chunks inside a byte range
    int64_t first = -1, last = -1;
    chunk_range(0, 10 << 20, 2 << 20, first, last);
    CHECK(first == 0 && last == 5);
    chunk_range(1 << 20, 10 << 20, 2 << 20, first, last);            // a partial first chunk is not inside
    CHECK(first == 1 && last == 5);
    chunk_range((2 << 20) + 1, (4 << 20) - 1, 2 << 20, first, last);  // no whole chunk: an empty range, never negative
    CHECK(first == 2 && last == 2);
    chunk_range(5, 9, 0, first, last);                                // a granularity of 0 counts as 1
    CHECK(first == 5 && last == 9);
    // give_slots: uniform 1 MiB slots, 2 MiB chunks, every chunk mapped; the K/V range starts at slot 100 (top)
    const uint64_t G = 2 << 20, slot_b = 1 << 20;
    auto off = [&](int64_t s) { return (uint64_t) s * slot_b; };
    std::vector<bool> is_mapped(64, true);
    auto mapped = [&](int64_t c) { return c >= 0 && c < (int64_t) is_mapped.size() && is_mapped[(size_t) c]; };
    int64_t gave = 0;
    CHECK(give_slots(100, 100, 10, 0, 0, G, off, mapped, gave) == 100 && gave == 0);   // nothing needed
    CHECK(give_slots(100, 100, 10, 3, 3, G, off, mapped, gave) == 100 && gave == 0);   // the spare covers it
    CHECK(give_slots(100, 100, 10, 3, 0, G, off, mapped, gave) == 94 && gave == 6);    // 3 chunks = 6 slots of 1 MiB
    CHECK(give_slots(94, 100, 10, 1, 0, G, off, mapped, gave) == 94 && gave == 0);     // still mapped: the 3 given up cover it
    for (int c = 47; c < 50; ++c) is_mapped[(size_t) c] = false;                        // kvg_ensure unmaps what it gave
    CHECK(give_slots(94, 100, 10, 1, 0, G, off, mapped, gave) == 92 && gave == 2);     // one more chunk: two more slots
    CHECK(give_slots(100, 100, 98, 3, 0, G, off, mapped, gave) == 98 && gave == 2);    // stops at the floor
    is_mapped.assign(64, true);
    is_mapped[48] = false;                                                              // an unmapped chunk does not count
    CHECK(give_slots(100, 100, 10, 2, 0, G, off, mapped, gave) == 94 && gave == 6);    // chunks 47 and 49 (48 was unmapped)
    if (fails == 0) std::puts("kv_grow_plan: OK");
    return fails == 0 ? 0 : 1;
}
