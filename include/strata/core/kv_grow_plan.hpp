// include/strata/core/kv_grow_plan.hpp - the elastic K/V's accounting for batch slots (--kv-grow-slots), host only.
//
// --kv-grow (generate.cpp, kvg_*) maps the K/V's 2 MiB chunks on demand, taking them from the expert cache's arena and
// giving them back when a request needs fewer cells.  With --kv-grow-slots the batch slots' sessions are elastic too:
// every state's pools grow together to the cells the furthest conversation reaches (qsa_kv_elastic_grow grows every
// pool to one count), and shrink only below what any slot still holds for its next turn.  This header is the arithmetic
// of that - what a window needs, what the slots hold, which chunks a slot range covers - kept pure so a host test
// pins it without a GPU.  Nothing here touches the device.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>

namespace strata::core::kvgrow {

/// Rounds `cells` up to a whole `step`, at most `max_context` (0 or fewer cells: 0).
inline int64_t round_cells(int64_t cells, int64_t step, int64_t max_context) {
    if (cells <= 0) return 0;
    step = std::max<int64_t>(step, 1);
    return std::min<int64_t>(max_context, (cells + step - 1) / step * step);
}

/// The cells the K/V must hold for a window whose `n` rows sit at `positions`: the furthest row plus the `rows_ahead`
/// cells the window may append after it and a `margin` (the solo path's 64: the drafter reads a little past the
/// window), rounded to `step`, at most `max_context`.  0 without rows.
inline int64_t target_cells(const int64_t* positions, int n, int64_t rows_ahead, int64_t margin, int64_t step,
                            int64_t max_context) {
    if (positions == nullptr || n <= 0) return 0;
    int64_t far = 0;
    for (int i = 0; i < n; ++i) far = std::max(far, positions[i]);
    return round_cells(far + rows_ahead + margin, step, max_context);
}

/// What the slots still hold between requests: a slot keeps its conversation (`held[b]` tokens) for that chat's next
/// turn, so the K/V may not shrink below the longest held plus `margin`.  0 when no slot holds anything.
inline int64_t slots_hold(const int64_t* held, int n, int64_t margin) {
    int64_t top = 0;
    for (int i = 0; i < n; ++i) top = std::max(top, held[i]);
    return top > 0 ? top + margin : 0;
}

/// The chunks (of `granularity` bytes) wholly inside the arena bytes [lo_bytes, top_bytes): [first, last).  kvg_ensure
/// may unmap exactly these when the expert slots in that byte range are given up (a chunk a live slot shares stays).
inline void chunk_range(uint64_t lo_bytes, uint64_t top_bytes, uint64_t granularity, int64_t& first, int64_t& last) {
    granularity = std::max<uint64_t>(granularity, 1);
    first = (int64_t) ((lo_bytes + granularity - 1) / granularity);
    last = (int64_t) (top_bytes / granularity);
    if (last < first) last = first;
}

/// kvg_ensure's loop, as arithmetic: starting from slot `lo` (the lowest slot already given up; `top` is the end of the
/// range the K/V may take), give up slots downwards (never below `floor`) until the mapped chunks wholly inside the
/// given-up range plus `spare` cover `need` chunks.  `slot_offset(s)` is slot s's first byte (s == top: the end),
/// `mapped(c)` whether chunk c is backed.  Returns the new `lo`; `gave` gets how many slots were given up.
inline int64_t give_slots(int64_t lo, int64_t top, int64_t floor, int64_t need, int64_t spare, uint64_t granularity,
                          const std::function<uint64_t(int64_t)>& slot_offset, const std::function<bool(int64_t)>& mapped,
                          int64_t& gave) {
    gave = 0;
    auto freeable = [&]() {
        int64_t first, last, k = 0;
        chunk_range(slot_offset(lo), slot_offset(top), granularity, first, last);
        for (int64_t c = first; c < last; ++c) k += mapped(c) ? 1 : 0;
        return k;
    };
    while (spare + freeable() < need && lo > floor) {
        --lo;
        ++gave;
    }
    return lo;
}

}  // namespace strata::core::kvgrow
