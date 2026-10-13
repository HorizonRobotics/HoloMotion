#pragma once
#include <cmath>
namespace holomotion {
inline bool fresh_policy_action(bool received, double age_seconds, double timeout_seconds) {
    return received && std::isfinite(age_seconds) && age_seconds >= 0.0 && age_seconds <= timeout_seconds;
}
inline bool default_transition_finished(double elapsed, double duration) {
    return std::isfinite(elapsed) && std::isfinite(duration) && duration >= 0.0 && elapsed >= duration;
}
}
