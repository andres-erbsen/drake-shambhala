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
  SimpleContinuousTimeSystem() : drake::systems::VectorSystem<T>(::drake::systems::SystemTypeTag<SimpleContinuousTimeSystem>{}, 2, 1) { // n_in, n_out
    this->DeclareContinuousState(4); // n_state
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

    // TODO: parametrize
    // DS Kinetic 60
    const double m = 2;
    const double g = 9.8;
    const double cD0 = .005;
    const double k = .08;
    const double S = .23;
    const double rho = 1; // TODO real value

    const auto& speed = state(0);
    const auto& pitch = state(1);
    const auto& yaw   = state(2);
    const auto& z     = state(3);

    const auto& cL    = input(0);
    const auto& roll  = input(1);

    const T windspeed_gradient = 0;

    const T altitude_dot = speed*sin(pitch);
    const T Wd = windspeed_gradient*altitude_dot;

    const auto cD = cD0 + k*cL*cL;
    const auto D = .5*cD*rho*S*speed*speed;
    const auto L = .5*cL*rho*S*speed*speed;

    (*derivatives)(0) = 1/(m                 )*(     -D      - m*g*sin(pitch) + m*Wd*cos(pitch)*sin(yaw));
    (*derivatives)(1) = 1/(m                 )*(     -D      - m*g*sin(pitch) + m*Wd*cos(pitch)*sin(yaw));
    (*derivatives)(2) = 1/(m*speed           )*( L*cos(roll) - m*g*cos(pitch) - m*Wd*sin(pitch)*sin(yaw));
    (*derivatives)(3) = 1/(m*speed*cos(pitch))*( L*sin(roll)                  + m*Wd           *cos(yaw));

    // notes:
    // ipopt needs speed replaced with (1e-4+speed), snopt only needs that if speed is not constrained away from 0

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

  // initial state
  dircol.AddLinearConstraint(dircol.initial_state()(0) == 13);
  dircol.AddLinearConstraint(dircol.initial_state()(1) == 0);
  dircol.AddLinearConstraint(dircol.initial_state()(3) == 0);

  // design limits
  dircol.AddConstraintToAllKnotPoints(dircol.state()(3) >= 0); // z >= 0

  dircol.AddConstraintToAllKnotPoints(dircol.input()(0) >= 0); // 0 <= cL <= 1.2
  dircol.AddConstraintToAllKnotPoints(dircol.input()(0) <= 1.2);

  // planning target
  dircol.AddFinalCost(-(9.8*dircol.state()(3) + .5*dircol.state()(0)*dircol.state()(0)) ); // negative energy

  // solver spoonfeeding
  dircol.AddConstraintToAllKnotPoints(dircol.state()(0) >= 13./2);

  auto result = dircol.Solve();
  if (result != drake::solvers::SolutionResult::kSolutionFound) {
    fprintf(stderr, "solving failed (%d)!\n", result);
    return 1;
  }

  {
    auto inputs = dircol.ReconstructInputTrajectory();
    auto traj = dircol.ReconstructStateTrajectory();
    auto timestamps = traj.get_segment_times();
    for (size_t i = 0; i < timestamps.size(); i++) {
      auto t = timestamps[i];
      auto u = inputs.value(timestamps[i]).coeff(0);
      auto x = traj.value(timestamps[i]).coeff(0);
      printf("%f < %f >   %f\n", t, u, x);
    }
    printf("\n");
  }


  return 0;
}
