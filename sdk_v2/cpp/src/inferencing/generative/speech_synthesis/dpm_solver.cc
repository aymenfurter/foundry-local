// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "inferencing/generative/speech_synthesis/dpm_solver.h"

#include "exception.h"

#include <cmath>

namespace fl {

std::vector<float> SampleDpmSolver(const DiffusionSchedule& schedule, const std::vector<float>& noise,
                                   const VelocityPredictor& predict) {
  const auto steps = schedule.timesteps.size();
  if (steps == 0 || schedule.sigmas.size() != steps + 1) {
    FL_THROW(FOUNDRY_LOCAL_ERROR_INVALID_ARGUMENT, "diffusion schedule needs one sigma more than timesteps");
  }

  std::vector<double> sample(noise.begin(), noise.end());
  std::vector<double> previous_x0;
  std::vector<float> input(noise.size());
  for (size_t step = 0; step < steps; ++step) {
    for (size_t i = 0; i < sample.size(); ++i) {
      input[i] = static_cast<float>(sample[i]);
    }

    const auto velocity = predict(step, input);
    if (velocity.size() != sample.size()) {
      FL_THROW(FOUNDRY_LOCAL_ERROR_INTERNAL, "diffusion model output has the wrong size");
    }

    // With alpha = 1 / sqrt(1 + sigma^2), the noise level is sigma * alpha and lambda = log(alpha / (sigma * alpha)).
    const double s0 = schedule.sigmas[step];
    const double alpha0 = 1.0 / std::sqrt(1.0 + s0 * s0);
    std::vector<double> x0(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
      x0[i] = alpha0 * sample[i] - s0 * alpha0 * static_cast<double>(velocity[i]);
    }

    const double s1 = schedule.sigmas[step + 1];
    if (s1 == 0.0) {
      sample = x0;
      previous_x0 = std::move(x0);
      continue;
    }

    const double alpha1 = 1.0 / std::sqrt(1.0 + s1 * s1);
    const double h = std::log(s0) - std::log(s1);
    const double ratio = (s1 * alpha1) / (s0 * alpha0);
    const double phi = alpha1 * std::expm1(-h);
    const bool first_order = step == 0 || step + 1 == steps || previous_x0.empty();
    const double r0 = first_order ? 1.0 : (std::log(schedule.sigmas[step - 1]) - std::log(s0)) / h;
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = ratio * sample[i] - phi * x0[i];
      if (!first_order) {
        sample[i] -= 0.5 * phi * (x0[i] - previous_x0[i]) / r0;
      }
    }

    previous_x0 = std::move(x0);
  }

  return std::vector<float>(sample.begin(), sample.end());
}

}  // namespace fl
