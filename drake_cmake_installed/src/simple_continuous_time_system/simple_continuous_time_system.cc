/*****************************************************************************
 * Copyright (c) 2017, Massachusetts Institute of Technology.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

/**
 * @file apps/simple_continuous_time_system.cc
 *
 * Simple Continuous Time System Example
 *
 * This is meant to be a sort of "hello world" example for the drake::system
 * classes. It defines a very simple continuous time system, simulates it from
 * a given initial condition, and plots the result.
 */

#include <cmath>
#include <cstdio>

#include <Eigen/Core>

#include <drake/common/autodiff.h>  // IWYU pragma: keep
#include <drake/common/default_scalars.h>
#include <drake/common/drake_assert.h>
#include <drake/common/unused.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/context.h>
#include <drake/systems/framework/continuous_state.h>
#include <drake/systems/framework/vector_system.h>
#include <drake/systems/trajectory_optimization/direct_collocation.h>

namespace shambhala {
namespace systems {

/**
 * Simple Continuous Time System
 *
 * xdot = -x + x^3
 * y = x
 */
template <typename T>
class SimpleContinuousTimeSystem final : public drake::systems::VectorSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimpleContinuousTimeSystem);
  SimpleContinuousTimeSystem() : drake::systems::VectorSystem<T>(::drake::systems::SystemTypeTag<SimpleContinuousTimeSystem>{}, 0, 1) { // n_in, n_out
    this->DeclareContinuousState(1); // n_state
  }

  // Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit SimpleContinuousTimeSystem(const SimpleContinuousTimeSystem<U>&) : SimpleContinuousTimeSystem<T>() {}

 private:
  // xdot = -x + x^3
  void DoCalcVectorTimeDerivatives(
      const drake::systems::Context<T>& context,
      const Eigen::VectorBlock<const drake::VectorX<T>>& input,
      const Eigen::VectorBlock<const drake::VectorX<T>>& state,
      Eigen::VectorBlock<drake::VectorX<T>>* derivatives) const override {
    drake::unused(context, input);
    (*derivatives)(0) = -state(0) + state(0)*state(0)*state(0);
  }

  // y = x
  void DoCalcVectorOutput(
      const drake::systems::Context<T>& context,
      const Eigen::VectorBlock<const drake::VectorX<T>>& input,
      const Eigen::VectorBlock<const drake::VectorX<T>>& state,
      Eigen::VectorBlock<drake::VectorX<T>>* output) const /*override*/ {
    drake::unused(context, input);
    *output = state;
  }
};

}  // namespace systems
}  // namespace shambhala

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::shambhala::systems::SimpleContinuousTimeSystem)

int main() {
  // Create the simple system.
  shambhala::systems::SimpleContinuousTimeSystem<double> system;

  auto context = system.CreateDefaultContext();
  const int N = 201;
  const double dt_min = 10./N;
  const double dt_max = 10./N;
  drake::systems::trajectory_optimization::DirectCollocation dircol(
      &system, *context, N, dt_min, dt_max);
  dircol.AddLinearConstraint(dircol.initial_state()(0) == .9);
  auto result = dircol.Solve();
  if (result != drake::solvers::SolutionResult::kSolutionFound) {
    fprintf(stderr, "solving failed (%d)!\n", result);
    return 1;
  }

  {
    // auto inputs = dircol.ReconstructInputTrajectory();
    auto traj = dircol.ReconstructStateTrajectory();
    auto timestamps = traj.get_segment_times();
    for (size_t i = 0; i < timestamps.size(); i++) {
      auto t = timestamps[i];
      auto x = traj.value(timestamps[i]).coeff(0);
      printf("%f    %f\n", t, x);
    }
    printf("\n");
    DRAKE_DEMAND(traj.value(timestamps[N-1]).coeff(0) < 1.0e-4);
  }


  return 0;
}
