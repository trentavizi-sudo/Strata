#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace strata::core {

struct BatchWindowSample {
    int64_t windows = 0;
    int64_t processed_rows = 0;  // rows run by the verifier, including pipeline padding and MTP rows
    int64_t active_rows = 0;     // live request slots represented by those windows
    int64_t elapsed_ms = 0;      // first included window's start to the last completion
};

// Count completed decode windows only. The caller emits one small protocol line per second,
// plus a final partial sample when all slots become idle; the default-disabled path emits none.
class BatchWindowSampler {
public:
    using Clock = std::chrono::steady_clock;

    explicit BatchWindowSampler(bool enabled) : enabled_(enabled) {}

    std::optional<BatchWindowSample> complete(int processed_rows, int active_rows, bool idle,
                                              Clock::time_point started, Clock::time_point now) {
        if (!enabled_ || processed_rows < 1 || active_rows < 1 || active_rows > processed_rows || now < started)
            return std::nullopt;
        if (pending_.windows == 0 || started < first_) first_ = started;
        ++pending_.windows;
        pending_.processed_rows += processed_rows;
        pending_.active_rows += active_rows;
        if (!idle && now - first_ < std::chrono::seconds(1)) return std::nullopt;
        BatchWindowSample sample = pending_;
        sample.elapsed_ms = std::max<int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(now - first_).count());
        pending_ = {};
        return sample;
    }

private:
    bool enabled_ = false;
    BatchWindowSample pending_{};
    Clock::time_point first_{};
};

/// STRATA_EXPERT_TELEMETRY=1: where a decode window's expert work went, as the engine's own counters say it.  One
/// `BXP` line about every second (and a final one when the slots go idle, or at the end of a solo request): the
/// windows and rows it covers, their wall span, the bytes the GPU read over the link for its PCIe share of the misses,
/// the host's time in the CPU pool and its wait for the GPU, and the routed entries served by each tier.  Every field
/// is a delta over the sample's windows, so a reader never has to know the engine's running totals.
struct ExpertWindowSample {
    int64_t windows = 0;
    int64_t rows = 0;
    int64_t elapsed_ms = 0;
    int64_t pcie_bytes = 0;      // the PCIe share's blob copies (expert_source.cpp: one blob per distinct expert)
    int64_t pool_us = 0;         // Verifier::ms_pool: the host inside the CPU pool (CUDA0's verifier)
    int64_t wait_us = 0;         // Verifier::ms_wait: the host waiting for the GPU to reach the layer
    int64_t link_est_us = 0;     // pcie_bytes at the start-up probe's host->device bandwidth: an ESTIMATE, not a measurement
    int64_t misses = 0;          // distinct (layer, expert) pairs the CPU computed
    int64_t hits = 0;            // routed entries the VRAM cache served
    int64_t entries = 0;         // routed entries in all (VRAM + CPU + PCIe/peer)
    int64_t offload = 0;         // routed entries served over PCIe or by another GPU
};

/// The sampler: `add` takes one window's deltas and says when a sample is due (a second has passed, or `idle`).
class ExpertWindowSampler {
public:
    using Clock = std::chrono::steady_clock;

    explicit ExpertWindowSampler(bool enabled) : enabled_(enabled) {}
    bool enabled() const { return enabled_; }

    std::optional<ExpertWindowSample> add(const ExpertWindowSample& w, bool idle, Clock::time_point started,
                                          Clock::time_point now) {
        if (!enabled_ || w.windows < 1 || now < started) return std::nullopt;
        if (pending_.windows == 0 || started < first_) first_ = started;
        pending_.windows += w.windows;
        pending_.rows += w.rows;
        pending_.pcie_bytes += w.pcie_bytes;
        pending_.pool_us += w.pool_us;
        pending_.wait_us += w.wait_us;
        pending_.link_est_us += w.link_est_us;
        pending_.misses += w.misses;
        pending_.hits += w.hits;
        pending_.entries += w.entries;
        pending_.offload += w.offload;
        if (!idle && now - first_ < std::chrono::seconds(1)) return std::nullopt;
        ExpertWindowSample sample = pending_;
        sample.elapsed_ms = std::max<int64_t>(1,
            std::chrono::duration_cast<std::chrono::milliseconds>(now - first_).count());
        pending_ = {};
        return sample;
    }

    /// The protocol line (without the newline): `BXP windows rows elapsed_ms pcie_bytes pool_us wait_us link_est_us
    /// misses hits entries offload`, integers only.
    static std::string line(const ExpertWindowSample& s) {
        char b[256];
        std::snprintf(b, sizeof b, "BXP %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld %lld",
                      (long long) s.windows, (long long) s.rows, (long long) s.elapsed_ms, (long long) s.pcie_bytes,
                      (long long) s.pool_us, (long long) s.wait_us, (long long) s.link_est_us, (long long) s.misses,
                      (long long) s.hits, (long long) s.entries, (long long) s.offload);
        return b;
    }

private:
    bool enabled_ = false;
    ExpertWindowSample pending_{};
    Clock::time_point first_{};
};

struct StageInFlightSample {
    int64_t elapsed_us = 0;
    std::vector<int64_t> in_flight_us;  // host time between a stage launch and its completed poll
};

// A pipeline stage holds at most one group. Measure the time it is reserved by
// engine work, including poll latency, without claiming GPU kernel utilization.
class StageInFlightSampler {
public:
    using Clock = std::chrono::steady_clock;

    StageInFlightSampler(int stages, bool enabled)
        : enabled_(enabled && stages > 1 && stages <= 16),
          active_(enabled_ ? (std::size_t) stages : 0),
          in_flight_us_(enabled_ ? (std::size_t) stages : 0, 0) {}

    void launch(int stage, Clock::time_point now) {
        if (!valid(stage)) return;
        if (!started_) started_ = now;
        active_[(std::size_t) stage] = now;
    }

    void complete(int stage, Clock::time_point now) {
        if (!valid(stage)) return;
        auto& since = active_[(std::size_t) stage];
        if (!since || now < *since) return;
        in_flight_us_[(std::size_t) stage] +=
            std::chrono::duration_cast<std::chrono::microseconds>(now - *since).count();
        since.reset();
    }

    std::optional<StageInFlightSample> sample(Clock::time_point now, bool idle) {
        if (!enabled_ || !started_ || now < *started_ ||
            (!idle && now - *started_ < std::chrono::seconds(1))) return std::nullopt;
        const auto elapsed = std::max<int64_t>(1,
            std::chrono::duration_cast<std::chrono::microseconds>(now - *started_).count());
        for (std::size_t stage = 0; stage < active_.size(); ++stage) {
            if (!active_[stage]) continue;
            in_flight_us_[stage] += std::chrono::duration_cast<std::chrono::microseconds>(
                now - *active_[stage]).count();
            active_[stage] = now;
        }
        StageInFlightSample result{elapsed, in_flight_us_};
        for (auto& value : result.in_flight_us) value = std::clamp(value, (int64_t) 0, elapsed);
        std::fill(in_flight_us_.begin(), in_flight_us_.end(), 0);
        if (idle) {
            started_.reset();
            std::fill(active_.begin(), active_.end(), std::nullopt);
        } else started_ = now;
        return result;
    }

private:
    bool valid(int stage) const { return enabled_ && stage >= 0 && (std::size_t) stage < active_.size(); }

    bool enabled_ = false;
    std::optional<Clock::time_point> started_;
    std::vector<std::optional<Clock::time_point>> active_;
    std::vector<int64_t> in_flight_us_;
};

}  // namespace strata::core
