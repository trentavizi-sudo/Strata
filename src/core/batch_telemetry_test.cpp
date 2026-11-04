#include "strata/core/batch_telemetry.hpp"

#include <chrono>
#include <cstdio>

int main() {
    using Sampler = strata::core::BatchWindowSampler;
    const auto t0 = Sampler::Clock::time_point{};
    Sampler disabled(false);
    if (disabled.complete(4, 2, true, t0, t0)) return 1;

    Sampler sampler(true);
    if (sampler.complete(4, 2, false, t0, t0)) return 2;
    if (sampler.complete(6, 3, false, t0 + std::chrono::milliseconds(490),
                         t0 + std::chrono::milliseconds(500))) return 3;
    const auto first = sampler.complete(8, 4, false, t0 + std::chrono::milliseconds(990),
                                        t0 + std::chrono::milliseconds(1000));
    if (!first || first->windows != 3 || first->processed_rows != 18 ||
        first->active_rows != 9 || first->elapsed_ms != 1000) return 4;
    if (sampler.complete(0, 0, true, t0, t0 + std::chrono::milliseconds(1001))) return 5;
    const auto last = sampler.complete(2, 1, true, t0 + std::chrono::milliseconds(1001),
                                       t0 + std::chrono::milliseconds(1002));
    if (!last || last->windows != 1 || last->processed_rows != 2 ||
        last->active_rows != 1 || last->elapsed_ms != 1) return 6;
    Sampler pipelined(true);
    if (pipelined.complete(4, 2, false, t0 + std::chrono::milliseconds(500),
                           t0 + std::chrono::milliseconds(900))) return 7;
    const auto overlap = pipelined.complete(4, 1, true, t0 + std::chrono::milliseconds(200),
                                            t0 + std::chrono::milliseconds(1000));
    if (!overlap || overlap->windows != 2 || overlap->processed_rows != 8 ||
        overlap->active_rows != 3 || overlap->elapsed_ms != 800) return 8;
    using StageSampler = strata::core::StageInFlightSampler;
    StageSampler stages(4, true);
    stages.launch(0, t0);
    stages.launch(1, t0 + std::chrono::milliseconds(200));
    stages.complete(0, t0 + std::chrono::milliseconds(800));
    if (stages.sample(t0 + std::chrono::milliseconds(999), false)) return 9;
    const auto first_stages = stages.sample(t0 + std::chrono::milliseconds(1000), false);
    if (!first_stages || first_stages->elapsed_us != 1000000 ||
        first_stages->in_flight_us != std::vector<int64_t>({800000, 800000, 0, 0})) return 10;
    stages.complete(1, t0 + std::chrono::milliseconds(1200));
    const auto last_stages = stages.sample(t0 + std::chrono::milliseconds(1200), true);
    if (!last_stages || last_stages->elapsed_us != 200000 ||
        last_stages->in_flight_us != std::vector<int64_t>({0, 200000, 0, 0})) return 11;
    StageSampler disabled_stages(4, false);
    disabled_stages.launch(0, t0);
    if (disabled_stages.sample(t0 + std::chrono::seconds(2), true)) return 12;
    using XSampler = strata::core::ExpertWindowSampler;
    using XSample = strata::core::ExpertWindowSample;
    XSampler xoff(false);
    if (xoff.add(XSample{1, 2, 0, 100, 10, 5, 2, 3, 4, 7, 0}, true, t0, t0)) return 13;
    XSampler xs(true);
    if (xs.add(XSample{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, false, t0, t0)) return 14;   // no window: nothing
    if (xs.add(XSample{1, 2, 0, 1000, 300, 100, 20, 5, 10, 20, 2}, false, t0, t0 + std::chrono::milliseconds(10))) return 15;
    const auto x1 = xs.add(XSample{1, 2, 0, 500, 200, 50, 10, 3, 12, 20, 1}, false, t0 + std::chrono::milliseconds(990),
                           t0 + std::chrono::milliseconds(1000));
    if (!x1 || x1->windows != 2 || x1->rows != 4 || x1->elapsed_ms != 1000 || x1->pcie_bytes != 1500 || x1->pool_us != 500 ||
        x1->wait_us != 150 || x1->link_est_us != 30 || x1->misses != 8 || x1->hits != 22 || x1->entries != 40 || x1->offload != 3)
        return 16;
    if (XSampler::line(*x1) != "BXP 2 4 1000 1500 500 150 30 8 22 40 3") return 17;
    const auto x2 = xs.add(XSample{1, 1, 0, 0, 7, 0, 0, 0, 9, 9, 0}, true, t0 + std::chrono::milliseconds(1001),
                           t0 + std::chrono::milliseconds(1001));   // idle: a partial sample at once, span at least 1 ms
    if (!x2 || x2->windows != 1 || x2->elapsed_ms != 1 || x2->pool_us != 7 || x2->hits != 9) return 18;
    std::puts("batch telemetry sampler: OK");
    return 0;
}
