// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "inferencing/generative/speech_synthesis/speech_synthesis_config.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace fl {

/// The model output (v-prediction) for `sample` at diffusion step `step`, with classifier-free guidance applied.
using VelocityPredictor = std::function<std::vector<float>(size_t step, const std::vector<float>& sample)>;

/// Denoise `noise` with multistep DPM-Solver++ for a v-prediction model: second order (midpoint) except for the first
/// and the last step (lower order final), as in diffusers' DPMSolverMultistepScheduler that VibeVoice uses with 5
/// steps. The state is kept in double precision. A final sigma of 0 returns the predicted clean sample.
std::vector<float> SampleDpmSolver(const DiffusionSchedule& schedule, const std::vector<float>& noise,
                                   const VelocityPredictor& predict);

}  // namespace fl
