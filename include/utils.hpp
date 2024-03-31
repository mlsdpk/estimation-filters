#pragma once

#include <matplot/matplot.h>

#include <Eigen/Dense>
#include <random>

namespace utils {

/**
 * @brief Generate random numbers from the multivariate normal distribution
 *
 */
inline Eigen::MatrixXd mvnrnd(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov,
                              const std::size_t n = 1u)
{
  static std::mt19937 gen{std::random_device{}()};
  static std::normal_distribution<> dist;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
  const Eigen::MatrixXd& transform =
      eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

  Eigen::MatrixXd output(mean.rows(), n);
  for (std::size_t i = 0; i < n; ++i)
  {
    output.col(i) = mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr(
                                           [&](auto x) { return dist(gen); });
  }

  return output;
}

/**
 * @brief Generates an N-long sequence of states using a Gaussian prior
 * and a linear Gaussian process model.
 *
 * @param x_0 Prior mean                   [n x 1]
 * @param P_0 Prior covariance             [n x n]
 * @param A   State transition matrix      [n x n]
 * @param Q   Process noise covariance     [n x n]
 * @param N   Number of states to generate [1 x 1]
 * @return    State vector sequence        [n x N+1]
 */
inline Eigen::VectorXd generateLinearStateSequence(const Eigen::VectorXd& x_0,
                                                   const Eigen::MatrixXd& P_0,
                                                   const Eigen::MatrixXd& A,
                                                   const Eigen::MatrixXd& Q, std::size_t N)
{
  // get the state dimensions
  const auto n = x_0.rows();

  // initialize output with all zeros
  Eigen::VectorXd output = Eigen::VectorXd::Zero(n, N);

  // sample initial state from the prior distribution x0 ~ N(x_0,P_0)
  output.col(0) = mvnrnd(x_0, P_0);

  // iterate to generate N samples
  for (std::size_t i = 0; i < N; ++N)
  {
    // Motion model: X{k} = A * X{k-1} + q{k-1}, where q{k-1} ~ N(0,Q)
    output.col(i + 1) = A * output.col(i) + mvnrnd(Eigen::VectorXd::Zero(n, 1), Q);
  }

  return output;
}

/**
 * @brief Generates x,y-points which lie on the ellipse describing
 * a sigma level in the Gaussian density defined by mean and covariance.
 *
 * @param mu      [2 x 1] Mean of the Gaussian density
 * @param sigma   [2 x 2] Covariance matrix of the Gaussian density
 * @param level   Which sigma level curve to plot. Can take any positive value,
 *                but common choices are 1, 2 or 3. Default = 3.
 * @param npoints Number of points on the ellipse to generate. Default = 32.
 */
inline auto sigmaEllipse2D(const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma,
                           const size_t level = 3u, const size_t npoints = 32)
{
  const auto& angles = matplot::linspace(0, 2 * M_PI, npoints);

  std::vector<double> x;
  std::vector<double> y;
  x.reserve(angles.size());
  y.reserve(angles.size());

  // Procedure:
  // - A 3 sigma level curve is given by {x} such that (x-mux)'*Q^-1*(x-mux) = 3^2
  //      or in scalar form: (x-mux) = sqrt(Q)*3
  // - replacing z= sqrtm(Q^-1)*(x-mux), such that we have now z'*z = 3^2
  //      which is now a circle with radius equal 3.
  // - Sampling in z, we have z = 3*[cos(theta); sin(theta)]', for theta=1:2*pi
  // - Back to x we get:  x = mux  + 3* sqrtm(Q)*[cos(theta); sin(theta)]'

  for (const auto ang : angles)
  {
    x.push_back(mu(0) + level * (sigma(0, 0) * std::cos(ang) + sigma(0, 1) * std::sin(ang)));
    y.push_back(mu(1) + level * (sigma(1, 0) * std::cos(ang) + sigma(1, 1) * std::sin(ang)));
  }

  return std::make_pair(x, y);
}

}  // namespace utils