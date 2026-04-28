#include "flightlib/common/command.hpp"


namespace flightlib {

Command::Command() {}

Command::Command(const Scalar t, const Scalar thrust, const Vector<3>& omega)
  : t(t), collective_thrust(thrust), omega(omega) {}

Command::Command(const Scalar t, const Vector<4>& thrusts)
  : t(t), thrusts(thrusts) {}

// Use a NaN check that survives -Ofast/-ffinite-math-only
static inline bool safe_isnan(float x) {
  volatile float v = x;
  return v != v;
}

static inline bool safe_isfinite(float x) {
  return !safe_isnan(x) && !safe_isnan(x - x);
}

static inline bool safe_vec3_finite(const Vector<3>& v) {
  return safe_isfinite(v(0)) && safe_isfinite(v(1)) && safe_isfinite(v(2));
}

static inline bool safe_vec4_finite(const Vector<4>& v) {
  return safe_isfinite(v(0)) && safe_isfinite(v(1)) && 
         safe_isfinite(v(2)) && safe_isfinite(v(3));
}

bool Command::valid() const {
  bool t_ok = safe_isfinite(t);
  bool rates_ok = safe_isfinite(collective_thrust) && safe_vec3_finite(omega);
  bool thrusts_ok = safe_vec4_finite(thrusts);
  // XOR: exactly one command mode should be active
  return t_ok && (rates_ok ^ thrusts_ok);
}

bool Command::isSingleRotorThrusts() const {
  return safe_isfinite(t) && safe_vec4_finite(thrusts);
}

bool Command::isRatesThrust() const {
  return safe_isfinite(t) && safe_isfinite(collective_thrust) &&
         safe_vec3_finite(omega);
}

}  // namespace flightlib
