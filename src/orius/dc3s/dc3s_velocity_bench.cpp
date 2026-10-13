#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

struct DC3SResult {
    double safe_u;
    double w_t;
    double h_cbf;
};

inline DC3SResult project_velocity_aware_dc3s(
    double pos, double vel, double hard_limit, double nominal_u,
    double latency_s, double min_force, double max_force,
    double robust_decel_floor, double safety_margin
) {
    double stale_excess = std::max(0.0, latency_s - 0.02);
    double w_t = std::clamp(std::exp(-3.2 * stale_excess), 0.05, 1.0);
    double tau_eff = latency_s + 0.35 * (1.0 - w_t) + 0.15;
    double dist_to_barrier = hard_limit - pos;
    double v_pos = std::max(0.0, vel);
    double d_stop = v_pos * tau_eff + (v_pos * v_pos) / (2.0 * robust_decel_floor) + safety_margin;
    double h_val = dist_to_barrier - d_stop;

    double safe_u = nominal_u;
    if (h_val <= 0.0 || w_t < 0.45) {
        safe_u = min_force;
    } else if (h_val < 12.0) {
        double alpha = h_val / 12.0;
        double req_decel = -((v_pos * v_pos) / (2.0 * std::max(1.0, dist_to_barrier - safety_margin)));
        safe_u = std::min(nominal_u * w_t, alpha * nominal_u + (1.0 - alpha) * std::max(min_force, req_decel * 1.35));
    } else if (nominal_u > 0.0) {
        safe_u = nominal_u * w_t;
    }
    safe_u = std::clamp(safe_u, min_force, max_force);
    return {safe_u, w_t, h_val};
}

int main() {
    const int N = 2000000;
    volatile double sink = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto res = project_velocity_aware_dc3s(35.0, 22.0, 85.0, 2.5, 0.02 + (i % 50) * 0.01, -8.0, 4.0, 2.4, 4.0);
        sink += res.safe_u;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_per_step = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    std::cout << "C++ O3 Velocity-Aware DC3S Mean Latency: " << us_per_step << " us/step (sink=" << sink << ")" << std::endl;
    return 0;
}
