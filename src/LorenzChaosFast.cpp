//============================================================================
// Name        : LorenzChaosFast.cpp
// Author      : Nathan Ormsby
// Version     :
// Copyright   : DO NOT COPY MY CODE, it probably wont work
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <tuple>
#include <random>
#include <iomanip>

using namespace std;
using namespace Eigen;

const double s = 10.0;
const double r = 28.0;
const double b = 8.0 / 3.0;
const double rescale_interval = 0.1;




// Utility functions

// Function to generate a random number between -1.0 and 1.0
double randDouble() {
    static uniform_real_distribution<double> distribution(-1.0, 1.0);
    static mt19937 generator;
    return distribution(generator);
}


void normalize(double& x, double& y, double& z) {
    double norm = sqrt(x * x + y * y + z * z);

    if (norm > 0) {
        x /= norm;
        y /= norm;
        z /= norm;
    }
}

void orthogonalize(double &dx1, double &dy1, double &dz1, double &dx2, double &dy2, double &dz2, double scale) {
    double dot_product = dx1 * dx2 + dy1 * dy2 + dz1 * dz2;
    double projection = dot_product / (scale * scale);

    dx2 -= projection * dx1;
    dy2 -= projection * dy1;
    dz2 -= projection * dz1;
}

void normalize2d(double& x, double& y) {
    double norm = sqrt(x * x + y * y);

    if (norm > 0) {
        x /= norm;
        y /= norm;
    }
}

void orthogonalize2d(double &dx1, double &dy1, double &dx2, double &dy2, double scale) {
    double dot_product = dx1 * dx2 + dy1 * dy2;
    double projection = dot_product / (scale * scale);

    dx2 -= projection * dx1;
    dy2 -= projection * dy1;
}

tuple<double, double, double> ortho_project_3_vectors(double x, double y, double z, double x1, double y1, double z1, double x2, double y2, double z2) {
    double proj1 = x * x1 + y * y1 + z * z1;
    double proj2 = x * x2 + y * y2 + z * z2;

    double xp = x - proj1 * x1 - proj2 * x2;
    double yp = y - proj1 * y1 - proj2 * y2;
    double zp = z - proj1 * z1 - proj2 * z2;

    return make_tuple(xp, yp, zp);
}

void orthogonalizeAndNormalize(double& x1, double& y1, double& z1, double& x2, double& y2, double& z2, double& x3, double& y3, double& z3) {
    // Normalize first vector
    normalize(x1, y1, z1);

    // Project out the first vector from the second vector and normalize
    tie(x2, y2, z2) = ortho_project_3_vectors(x2, y2, z2, x1, y1, z1, x3, y3, z3);
    normalize(x2, y2, z2);

    // Project out the first and second vectors from the third vector and normalize
    tie(x3, y3, z3) = ortho_project_3_vectors(x3, y3, z3, x1, y1, z1, x2, y2, z2);
    normalize(x3, y3, z3);
}

// Initial condition function

vector<tuple<double, double, double>> generate_even_initial_conditions(int n_points, double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) {
    double x_step = (x_max - x_min) / std::ceil(std::pow(n_points, 1.0/3));
    double y_step = (y_max - y_min) / std::ceil(std::pow(n_points, 1.0/3));
    double z_step = (z_max - z_min) / std::ceil(std::pow(n_points, 1.0/3));

    vector<tuple<double, double, double>> initial_conditions;
    for (double x = x_min; x <= x_max; x += x_step) {
        for (double y = y_min; y <= y_max; y += y_step) {
            for (double z = z_min; z <= z_max; z += z_step) {
                initial_conditions.push_back(make_tuple(x, y, z));
                if (initial_conditions.size() == n_points) {
                    return initial_conditions;
                }
            }
        }
    }
    return initial_conditions;
}

vector<tuple<double, double>> generate_even_initial_conditions_2d(int n_points, double x_min, double x_max, double y_min, double y_max) {
    double x_step = (x_max - x_min) / std::ceil(std::sqrt(n_points));
    double y_step = (y_max - y_min) / std::ceil(std::sqrt(n_points));

    vector<tuple<double, double>> initial_conditions;
    for (double x = x_min; x <= x_max; x += x_step) {
        for (double y = y_min; y <= y_max; y += y_step) {
            initial_conditions.push_back(make_tuple(x, y));
            if (initial_conditions.size() == n_points) {
                return initial_conditions;
            }
        }
    }
    return initial_conditions;
}

// LORENZ
tuple<double, double, double> lorenz(double x, double y, double z) {
    double dxdt = s * (y - x);
    double dydt = r * x - y - x * z;
    double dzdt = x * y - b * z;
    return make_tuple(dxdt, dydt, dzdt);
}

tuple<double, double, double> rk4_lorenz(double x, double y, double z, double dt) {
    auto k1 = lorenz(x, y, z);
    auto k2 = lorenz(x + 0.5 * dt * get<0>(k1), y + 0.5 * dt * get<1>(k1), z + 0.5 * dt * get<2>(k1));
    auto k3 = lorenz(x + 0.5 * dt * get<0>(k2), y + 0.5 * dt * get<1>(k2), z + 0.5 * dt * get<2>(k2));
    auto k4 = lorenz(x + dt * get<0>(k3), y + dt * get<1>(k3), z + dt * get<2>(k3));

    double xn = x + dt * (get<0>(k1) + 2 * get<0>(k2) + 2 * get<0>(k3) + get<0>(k4)) / 6;
    double yn = y + dt * (get<1>(k1) + 2 * get<1>(k2) + 2 * get<1>(k3) + get<1>(k4)) / 6;
    double zn = z + dt * (get<2>(k1) + 2 * get<2>(k2) + 2 * get<2>(k3) + get<2>(k4)) / 6;

    return make_tuple(xn, yn, zn);
}

tuple<double, double, double> wolf_method_lorenz(double x0, double y0, double z0, double tf, double dt, double scale) {
    int n = static_cast<int>((tf - 20.0) / dt);
    double x = x0;
    double y = y0;
    double z = z0;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;
    double lyapunov_sum3 = 0;

    // Perturb
    double x_perturbed1 = x + scale;
    double y_perturbed1 = y;
    double z_perturbed1 = z;

    double x_perturbed2 = x;
    double y_perturbed2 = y + scale;
    double z_perturbed2 = z;

    double x_perturbed3 = x;
    double y_perturbed3 = y;
    double z_perturbed3 = z + scale;

    // Ignore the first 20 seconds of the simulation
    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
        tie(x, y, z) = rk4_lorenz(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_lorenz(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_lorenz(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_lorenz(x_perturbed3, y_perturbed3, z_perturbed3, dt);
    }

    for (int i = 0; i < n; ++i) {
        tie(x, y, z) = rk4_lorenz(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_lorenz(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_lorenz(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_lorenz(x_perturbed3, y_perturbed3, z_perturbed3, dt);

        double dx1 = x_perturbed1 - x;
        double dy1 = y_perturbed1 - y;
        double dz1 = z_perturbed1 - z;

        // V1
        // Get magnitude
        double distance1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        // Add to sum
		lyapunov_sum1 += log(distance1 / scale);

		// Renormalize and scale
		normalize(dx1, dy1, dz1);
		dx1 *= scale;
		dy1 *= scale;
		dz1 *= scale;

		// V2

		// Pull back v2 from v1
		double dx2 = x_perturbed2 - x;
		double dy2 = y_perturbed2 - y;
		double dz2 = z_perturbed2 - z;
		orthogonalize(dx1, dy1, dz1, dx2, dy2, dz2, scale);

		// Get magnitude
		double distance2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

		// Add to sum
		lyapunov_sum2 += log(distance2 / scale);

		// Renormalize and scale
		normalize(dx2, dy2, dz2);
		dx2 *= scale;
		dy2 *= scale;
		dz2 *= scale;

		// V3
		// Pull back from both v1 and v2
		double dx3 = x_perturbed3 - x;
		double dy3 = y_perturbed3 - y;
		double dz3 = z_perturbed3 - z;

		orthogonalize(dx1, dy1, dz1, dx3, dy3, dz3, scale);
		orthogonalize(dx2, dy2, dz2, dx3, dy3, dz3, scale);

		// Get magnitude
		double distance3 = sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

		// Add to sum
		lyapunov_sum3 += log(distance3 / scale);

		// Renormalize and scale
		normalize(dx3, dy3, dz3);
		dx3 *= scale;
		dy3 *= scale;
		dz3 *= scale;

		// Rescale the perturbed points
		x_perturbed1 = x + dx1;
		y_perturbed1 = y + dy1;
		z_perturbed1 = z + dz1;

		x_perturbed2 = x + dx2;
		y_perturbed2 = y + dy2;
		z_perturbed2 = z + dz2;

		x_perturbed3 = x + dx3;
		y_perturbed3 = y + dy3;
		z_perturbed3 = z + dz3;

    }
    double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
    double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);
    double lyapunov_exponent3 = lyapunov_sum3 / (n * dt);

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}



// Result calling functions
tuple<double, double, double> multiple_lorenz_results(vector<tuple<double, double, double>> initial_conditions, double tf, double dt, double lorenz_scale) {
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;

    int i = 0;
    for (auto& initial_condition : initial_conditions) {
    	if (i % 5  == 0) {
    		cout << "i: " << i << endl;
    	}
        tuple<double, double, double> lyapunov_exponents = wolf_method_lorenz(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, lorenz_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);

        ++i;
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();
    double lyapunov_exponent3 = lyapunov_sum3 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

tuple<double, double, double> single_lorenz_results(double x0, double y0, double z0, double tf, double dt, double lorenz_scale) {
    vector<tuple<double, double, double>> initial_conditions = {make_tuple(x0, y0, z0)};
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;

    for (auto& initial_condition : initial_conditions) {
        tuple<double, double, double> lyapunov_exponents = wolf_method_lorenz(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, lorenz_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();
    double lyapunov_exponent3 = lyapunov_sum3 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}


// ROSSLER

tuple<double, double, double> rossler(double x, double y, double z) {
    const double a = 0.2;
    const double b = 0.2;
    const double c = 5.7;

    double dxdt = -y - z;
    double dydt = x + a * y;
    double dzdt = b + z * (x - c);

    return make_tuple(dxdt, dydt, dzdt);
}

tuple<double, double, double> rk4_rossler(double x, double y, double z, double dt) {
    auto k1 = rossler(x, y, z);
    auto k2 = rossler(x + 0.5 * dt * get<0>(k1), y + 0.5 * dt * get<1>(k1), z + 0.5 * dt * get<2>(k1));
    auto k3 = rossler(x + 0.5 * dt * get<0>(k2), y + 0.5 * dt * get<1>(k2), z + 0.5 * dt * get<2>(k2));
    auto k4 = rossler(x + dt * get<0>(k3), y + dt * get<1>(k3), z + dt * get<2>(k3));

    double xn = x + dt * (get<0>(k1) + 2 * get<0>(k2) + 2 * get<0>(k3) + get<0>(k4)) / 6;
    double yn = y + dt * (get<1>(k1) + 2 * get<1>(k2) + 2 * get<1>(k3) + get<1>(k4)) / 6;
    double zn = z + dt * (get<2>(k1) + 2 * get<2>(k2) + 2 * get<2>(k3) + get<2>(k4)) / 6;

    return make_tuple(xn, yn, zn);
}

tuple<double, double, double> wolf_method_rossler(double x0, double y0, double z0, double tf, double dt, double scale) {
    int n = static_cast<int>((tf - 20.0) / dt);
    double x = x0;
    double y = y0;
    double z = z0;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;
    double lyapunov_sum3 = 0;

    // Perturb
    double x_perturbed1 = x + scale;
    double y_perturbed1 = y;
    double z_perturbed1 = z;

    double x_perturbed2 = x;
    double y_perturbed2 = y + scale;
    double z_perturbed2 = z;

    double x_perturbed3 = x;
    double y_perturbed3 = y;
    double z_perturbed3 = z + scale;

    // Ignore the first 20 seconds of the simulation
    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
        tie(x, y, z) = rk4_rossler(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_rossler(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_rossler(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_rossler(x_perturbed3, y_perturbed3, z_perturbed3, dt);
    }

    for (int i = 0; i < n; ++i) {
        tie(x, y, z) = rk4_rossler(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_rossler(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_rossler(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_rossler(x_perturbed3, y_perturbed3, z_perturbed3, dt);

        double dx1 = x_perturbed1 - x;
        double dy1 = y_perturbed1 - y;
        double dz1 = z_perturbed1 - z;

        // V1
        // Get magnitude
        double distance1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        // Add to sum
		lyapunov_sum1 += log(distance1 / scale);

		// Renormalize and scale
		normalize(dx1, dy1, dz1);
		dx1 *= scale;
		dy1 *= scale;
		dz1 *= scale;

		// V2

		// Pull back v2 from v1
		double dx2 = x_perturbed2 - x;
		double dy2 = y_perturbed2 - y;
		double dz2 = z_perturbed2 - z;
		orthogonalize(dx1, dy1, dz1, dx2, dy2, dz2, scale);

		// Get magnitude
		double distance2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

		// Add to sum
		lyapunov_sum2 += log(distance2 / scale);

		// Renormalize and scale
		normalize(dx2, dy2, dz2);
		dx2 *= scale;
		dy2 *= scale;
		dz2 *= scale;

		// V3
		// Pull back from both v1 and v2
		double dx3 = x_perturbed3 - x;
		double dy3 = y_perturbed3 - y;
		double dz3 = z_perturbed3 - z;

		orthogonalize(dx1, dy1, dz1, dx3, dy3, dz3, scale);
		orthogonalize(dx2, dy2, dz2, dx3, dy3, dz3, scale);

		// Get magnitude
		double distance3 = sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

		// Add to sum
		lyapunov_sum3 += log(distance3 / scale);

		// Renormalize and scale
		normalize(dx3, dy3, dz3);
		dx3 *= scale;
		dy3 *= scale;
		dz3 *= scale;

		// Rescale the perturbed points
		x_perturbed1 = x + dx1;
		y_perturbed1 = y + dy1;
		z_perturbed1 = z + dz1;

		x_perturbed2 = x + dx2;
		y_perturbed2 = y + dy2;
		z_perturbed2 = z + dz2;

		x_perturbed3 = x + dx3;
		y_perturbed3 = y + dy3;
		z_perturbed3 = z + dz3;

    }
    double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
    double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);
    double lyapunov_exponent3 = lyapunov_sum3 / (n * dt);

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

// Reporting results

tuple<double, double, double> multiple_rossler_results(vector<tuple<double, double, double>> initial_conditions, double tf, double dt, double rossler_scale) {
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;

    int i = 0;
    for (auto& initial_condition : initial_conditions) {
        if (i % 5 == 0) {
            cout << "i: " << i << endl;
        }
        tuple<double, double, double> lyapunov_exponents = wolf_method_rossler(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, rossler_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);

        ++i;
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();
    double lyapunov_exponent3 = lyapunov_sum3 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

tuple<double, double, double> single_rossler_results(double x0, double y0, double z0, double tf, double dt, double rossler_scale) {
    vector<tuple<double, double, double>> initial_conditions = {make_tuple(x0, y0, z0)};
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;
    for (auto& initial_condition : initial_conditions) {
        tuple<double, double, double> lyapunov_exponents = wolf_method_rossler(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, rossler_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);
    }

    double lyapunov_exponent1 = lyapunov_sum1;
    double lyapunov_exponent2 = lyapunov_sum2;
    double lyapunov_exponent3 = lyapunov_sum3;

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

// rabinovich

tuple<double, double, double> rabinovich_fabrikant(double x, double y, double z, double a, double b) {
    double dxdt = y * (z - 1 + x * x) + a * x;
    double dydt = x * (3 * z + 1 - x * x) + a * y;
    double dzdt = -2 * z * (b + x * y);

    return make_tuple(dxdt, dydt, dzdt);
}

tuple<double, double, double> rk4_rabinovich_fabrikant(double x, double y, double z, double a, double b, double dt) {
    auto k1 = rabinovich_fabrikant(x, y, z, a, b);
    auto k2 = rabinovich_fabrikant(x + 0.5 * dt * get<0>(k1), y + 0.5 * dt * get<1>(k1), z + 0.5 * dt * get<2>(k1), a, b);
    auto k3 = rabinovich_fabrikant(x + 0.5 * dt * get<0>(k2), y + 0.5 * dt * get<1>(k2), z + 0.5 * dt * get<2>(k2), a, b);
    auto k4 = rabinovich_fabrikant(x + dt * get<0>(k3), y + dt * get<1>(k3), z + dt * get<2>(k3), a, b);

    double xn = x + dt * (get<0>(k1) + 2 * get<0>(k2) + 2 * get<0>(k3) + get<0>(k4)) / 6;
    double yn = y + dt * (get<1>(k1) + 2 * get<1>(k2) + 2 * get<1>(k3) + get<1>(k4)) / 6;
    double zn = z + dt * (get<2>(k1) + 2 * get<2>(k2) + 2 * get<2>(k3) + get<2>(k4)) / 6;

    return make_tuple(xn, yn, zn);
}

tuple<double, double, double> wolf_method_rabinovich_fabrikant(double x0, double y0, double z0, double tf, double dt, double a, double b, double scale) {
    int n = static_cast<int>((tf - 20.0) / dt);
    double x = x0;
    double y = y0;
    double z = z0;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;
    double lyapunov_sum3 = 0;

    // Perturb
    double x_perturbed1 = x + scale;
    double y_perturbed1 = y;
    double z_perturbed1 = z;

    double x_perturbed2 = x;
    double y_perturbed2 = y + scale;
    double z_perturbed2 = z;

    double x_perturbed3 = x;
    double y_perturbed3 = y;
    double z_perturbed3 = z + scale;

    // Ignore the first 20 seconds of the simulation
    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
        tie(x, y, z) = rk4_rabinovich_fabrikant(x, y, z, a, b, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_rabinovich_fabrikant(x_perturbed1, y_perturbed1, z_perturbed1, a, b, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_rabinovich_fabrikant(x_perturbed2, y_perturbed2, z_perturbed2, a, b, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_rabinovich_fabrikant(x_perturbed3, y_perturbed3, z_perturbed3, a, b, dt);
    }

    for (int i = 0; i < n; ++i) {
        tie(x, y, z) = rk4_rabinovich_fabrikant(x, y, z, a, b, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_rabinovich_fabrikant(x_perturbed1, y_perturbed1, z_perturbed1, a, b, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_rabinovich_fabrikant(x_perturbed2, y_perturbed2, z_perturbed2, a, b, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_rabinovich_fabrikant(x_perturbed3, y_perturbed3, z_perturbed3, a, b, dt);

        double dx1 = x_perturbed1 - x;
        double dy1 = y_perturbed1 - y;
        double dz1 = z_perturbed1 - z;

        // V1
        // Get magnitude
        double distance1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        // Add to sum
		lyapunov_sum1 += log(distance1 / scale);

		// Renormalize and scale
		normalize(dx1, dy1, dz1);
		dx1 *= scale;
		dy1 *= scale;
		dz1 *= scale;

		// V2

		// Pull back v2 from v1
		double dx2 = x_perturbed2 - x;
		double dy2 = y_perturbed2 - y;
		double dz2 = z_perturbed2 - z;
		orthogonalize(dx1, dy1, dz1, dx2, dy2, dz2, scale);

		// Get magnitude
		double distance2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

		// Add to sum
		lyapunov_sum2 += log(distance2 / scale);

		// Renormalize and scale
		normalize(dx2, dy2, dz2);
		dx2 *= scale;
		dy2 *= scale;
		dz2 *= scale;

		// V3
		// Pull back from both v1 and v2
		double dx3 = x_perturbed3 - x;
		double dy3 = y_perturbed3 - y;
		double dz3 = z_perturbed3 - z;

		orthogonalize(dx1, dy1, dz1, dx3, dy3, dz3, scale);
		orthogonalize(dx2, dy2, dz2, dx3, dy3, dz3, scale);

		// Get magnitude
		double distance3 = sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

		// Add to sum
		lyapunov_sum3 += log(distance3 / scale);

		// Renormalize and scale
		normalize(dx3, dy3, dz3);
		dx3 *= scale;
		dy3 *= scale;
		dz3 *= scale;

		// Rescale the perturbed points
		x_perturbed1 = x + dx1;
		y_perturbed1 = y + dy1;
		z_perturbed1 = z + dz1;

		x_perturbed2 = x + dx2;
		y_perturbed2 = y + dy2;
		z_perturbed2 = z + dz2;

		x_perturbed3 = x + dx3;
		y_perturbed3 = y + dy3;
		z_perturbed3 = z + dz3;

    }
    double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
    double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);
    double lyapunov_exponent3 = lyapunov_sum3 / (n * dt);

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

// Reporting results

tuple<double, double, double> multiple_rabinovich_fabrikant_results(vector<tuple<double, double, double>> initial_conditions, double tf, double dt, double rabinovich_scale, double a, double b) {
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;

    int i = 0;
    for (auto& initial_condition : initial_conditions) {
        if (i % 5 == 0) {
            cout << "i: " << i << endl;
        }
        tuple<double, double, double> lyapunov_exponents = wolf_method_rabinovich_fabrikant(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, a, b, rabinovich_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);

        ++i;
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();
    double lyapunov_exponent3 = lyapunov_sum3 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

tuple<double, double, double> single_rabinovich_fabrikant_results(double x0, double y0, double z0, double tf, double dt, double a, double b, double scale) {
    vector<tuple<double, double, double>> initial_conditions = {make_tuple(x0, y0, z0)};
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;
    for (auto& initial_condition : initial_conditions) {
        tuple<double, double, double> lyapunov_exponents = wolf_method_rabinovich_fabrikant(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, a, b, scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);
    }

    double lyapunov_exponent1 = lyapunov_sum1;
    double lyapunov_exponent2 = lyapunov_sum2;
    double lyapunov_exponent3 = lyapunov_sum3;

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}


// Chua circuit:
tuple<double, double, double> chua(double x, double y, double z, double alpha=9, double beta=100.0/7.0, double a=-8.0/7.0, double b=-5.0/7.0) {
    double dxdt = alpha * (y - x - a * x * x);
    double dydt = x - y + z;
    double dzdt = -beta * y - b * z;
    return make_tuple(dxdt, dydt, dzdt);
}

tuple<double, double, double> rk4_chua(double x, double y, double z, double dt, double alpha=9, double beta=100.0/7.0, double a=-8.0/7.0, double b=-5.0/7.0) {
    auto k1 = chua(x, y, z, alpha, beta, a, b);
    auto k2 = chua(x + 0.5 * dt * get<0>(k1), y + 0.5 * dt * get<1>(k1), z + 0.5 * dt * get<2>(k1), alpha, beta, a, b);
    auto k3 = chua(x + 0.5 * dt * get<0>(k2), y + 0.5 * dt * get<1>(k2), z + 0.5 * dt * get<2>(k2), alpha, beta, a, b);
    auto k4 = chua(x + dt * get<0>(k3), y + dt * get<1>(k3), z + dt * get<2>(k3), alpha, beta, a, b);

    double xn = x + dt * (get<0>(k1) + 2 * get<0>(k2) + 2 * get<0>(k3) + get<0>(k4)) / 6;
    double yn = y + dt * (get<1>(k1) + 2 * get<1>(k2) + 2 * get<1>(k3) + get<1>(k4)) / 6;
    double zn = z + dt * (get<2>(k1) + 2 * get<2>(k2) + 2 * get<2>(k3) + get<2>(k4)) / 6;

    return make_tuple(xn, yn, zn);
}

tuple<double, double, double> wolf_method_chua(double x0, double y0, double z0, double tf, double dt, double scale) {
    int n = static_cast<int>((tf - 20.0) / dt);
    double x = x0;
    double y = y0;
    double z = z0;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;
    double lyapunov_sum3 = 0;

    // Perturb
    double x_perturbed1 = x + scale;
    double y_perturbed1 = y;
    double z_perturbed1 = z;

    double x_perturbed2 = x;
    double y_perturbed2 = y + scale;
    double z_perturbed2 = z;

    double x_perturbed3 = x;
    double y_perturbed3 = y;
    double z_perturbed3 = z + scale;

    // Ignore the first 20 seconds of the simulation
    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
        tie(x, y, z) = rk4_chua(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_chua(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_chua(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_chua(x_perturbed3, y_perturbed3, z_perturbed3, dt);
    }

    for (int i = 0; i < n; ++i) {
        tie(x, y, z) = rk4_chua(x, y, z, dt);
        tie(x_perturbed1, y_perturbed1, z_perturbed1) = rk4_chua(x_perturbed1, y_perturbed1, z_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2, z_perturbed2) = rk4_chua(x_perturbed2, y_perturbed2, z_perturbed2, dt);
        tie(x_perturbed3, y_perturbed3, z_perturbed3) = rk4_chua(x_perturbed3, y_perturbed3, z_perturbed3, dt);

        double dx1 = x_perturbed1 - x;
        double dy1 = y_perturbed1 - y;
        double dz1 = z_perturbed1 - z;

        // V1
        // Get magnitude
        double distance1 = sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

        // Add to sum
        lyapunov_sum1 += log(distance1 / scale);

        // Renormalize and scale
        normalize(dx1, dy1, dz1);
        dx1 *= scale;
        dy1 *= scale;
        dz1 *= scale;

        // V2

        // Pull back v2 from v1
        double dx2 = x_perturbed2 - x;
        double dy2 = y_perturbed2 - y;
        double dz2 = z_perturbed2 - z;
		orthogonalize(dx1, dy1, dz1, dx2, dy2, dz2, scale);

		// Get magnitude
		double distance2 = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

		// Add to sum
		lyapunov_sum2 += log(distance2 / scale);

		// Renormalize and scale
		normalize(dx2, dy2, dz2);
		dx2 *= scale;
		dy2 *= scale;
		dz2 *= scale;

		// V3
		// Pull back from both v1 and v2
		double dx3 = x_perturbed3 - x;
		double dy3 = y_perturbed3 - y;
		double dz3 = z_perturbed3 - z;

		orthogonalize(dx1, dy1, dz1, dx3, dy3, dz3, scale);
		orthogonalize(dx2, dy2, dz2, dx3, dy3, dz3, scale);

		// Get magnitude
		double distance3 = sqrt(dx3 * dx3 + dy3 * dy3 + dz3 * dz3);

		// Add to sum
		lyapunov_sum3 += log(distance3 / scale);

		// Renormalize and scale
		normalize(dx3, dy3, dz3);
		dx3 *= scale;
		dy3 *= scale;
		dz3 *= scale;

		// Rescale the perturbed points
		x_perturbed1 = x + dx1;
		y_perturbed1 = y + dy1;
		z_perturbed1 = z + dz1;

		x_perturbed2 = x + dx2;
		y_perturbed2 = y + dy2;
		z_perturbed2 = z + dz2;

		x_perturbed3 = x + dx3;
		y_perturbed3 = y + dy3;
		z_perturbed3 = z + dz3;
	}

	double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
	double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);
	double lyapunov_exponent3 = lyapunov_sum3 / (n * dt);

	return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

// Reporting results
tuple<double, double, double> multiple_chua_results(vector<tuple<double, double, double>> initial_conditions, double tf, double dt, double chua_scale) {
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;

    int i = 0;
    for (auto& initial_condition : initial_conditions) {
        if (i % 5 == 0) {
            cout << "i: " << i << endl;
        }
        tuple<double, double, double> lyapunov_exponents = wolf_method_chua(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, chua_scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);

        ++i;
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();
    double lyapunov_exponent3 = lyapunov_sum3 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

tuple<double, double, double> single_chua_results(double x0, double y0, double z0, double tf, double dt, double scale) {
    vector<tuple<double, double, double>> initial_conditions = {make_tuple(x0, y0, z0)};
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0, lyapunov_sum3 = 0;
    for (auto& initial_condition : initial_conditions) {
        tuple<double, double, double> lyapunov_exponents = wolf_method_chua(get<0>(initial_condition), get<1>(initial_condition), get<2>(initial_condition), tf, dt, scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
        lyapunov_sum3 += get<2>(lyapunov_exponents);
    }

    double lyapunov_exponent1 = lyapunov_sum1;
    double lyapunov_exponent2 = lyapunov_sum2;
    double lyapunov_exponent3 = lyapunov_sum3;

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;
    cout << "  Lambda 3 = " << lyapunov_exponent3 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}

// Henon map:

tuple<double, double> henon(double x, double y) {
    static const double a = 1.4;
    static const double b = 0.3;
    double xn = y + 1 - a * x * x;
    double yn = b * x;
    return make_tuple(xn, yn);
}

tuple<double, double> rk4_henon(double x, double y, double dt) {
    auto k1 = henon(x, y);
    auto k2 = henon(x + 0.5 * dt * get<0>(k1), y + 0.5 * dt * get<1>(k1));
    auto k3 = henon(x + 0.5 * dt * get<0>(k2), y + 0.5 * dt * get<1>(k2));
    auto k4 = henon(x + dt * get<0>(k3), y + dt * get<1>(k3));

    double xn = x + dt * (get<0>(k1) + 2 * get<0>(k2) + 2 * get<0>(k3) + get<0>(k4)) / 6;
    double yn = y + dt * (get<1>(k1) + 2 * get<1>(k2) + 2 * get<1>(k3) + get<1>(k4)) / 6;

    return make_tuple(xn, yn);
}

tuple<double, double> wolf_method_henon(double x0, double y0, double tf, double dt, double scale) {
    int n = static_cast<int>((tf - 20.0) / dt);
    double x = x0;
    double y = y0;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;

    // Perturb
    double x_perturbed1 = x + scale;
    double y_perturbed1 = y;

    double x_perturbed2 = x;
    double y_perturbed2 = y + scale;

    // Ignore the first 20 seconds of the simulation
    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
        tie(x, y) = rk4_henon(x, y, dt);
        tie(x_perturbed1, y_perturbed1) = rk4_henon(x_perturbed1, y_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2) = rk4_henon(x_perturbed2, y_perturbed2, dt);
    }

    for (int i = 0; i < n; ++i) {
        tie(x, y) = rk4_henon(x, y, dt);
        tie(x_perturbed1, y_perturbed1) = rk4_henon(x_perturbed1, y_perturbed1, dt);
        tie(x_perturbed2, y_perturbed2) = rk4_henon(x_perturbed2, y_perturbed2, dt);

        double dx1 = x_perturbed1 - x;
        double dy1 = y_perturbed1 - y;

        // V1
        // Get magnitude
        double distance1 = sqrt(dx1 * dx1 + dy1 * dy1);

        // Add to sum
        lyapunov_sum1 += log(distance1 / scale);

        // Renormalize and scale
        normalize2d(dx1, dy1);
        dx1 *= scale;
        dy1 *= scale;

        // V2

        // Pull back v2 from v1
        double dx2 = x_perturbed2 - x;
        double dy2 = y_perturbed2 - y;
        orthogonalize2d(dx1, dy1, dx2, dy2, scale);

        // Get magnitude
        double distance2 = sqrt(dx2 * dx2 + dy2 * dy2);

        // Add to sum
        lyapunov_sum2 += log(distance2 / scale);

        // Renormalize and scale
        normalize2d(dx2, dy2);
        dx2 *= scale;
        dy2 *= scale;
        // Rescale the perturbed points
		x_perturbed1 = x + dx1;
		y_perturbed1 = y + dy1;

		x_perturbed2 = x + dx2;
		y_perturbed2 = y + dy2;
	}

	double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
	double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);

	return make_tuple(lyapunov_exponent1, lyapunov_exponent2);
}

// Result reporting

tuple<double, double> multiple_henon_results(vector<tuple<double, double>> initial_conditions, double tf, double dt, double scale) {
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0;

    int i = 0;
    for (auto& initial_condition : initial_conditions) {
        if (i % 5 == 0) {
            cout << "i: " << i << endl;
        }
        tuple<double, double> lyapunov_exponents = wolf_method_henon(get<0>(initial_condition), get<1>(initial_condition), tf, dt, scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);

        ++i;
    }

    double lyapunov_exponent1 = lyapunov_sum1 / initial_conditions.size();
    double lyapunov_exponent2 = lyapunov_sum2 / initial_conditions.size();

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2);
}

tuple<double, double> single_henon_results(double x0, double y0, double tf, double dt, double scale) {
    vector<tuple<double, double>> initial_conditions = {make_tuple(x0, y0)};
    double lyapunov_sum1 = 0, lyapunov_sum2 = 0;

    for (auto& initial_condition : initial_conditions) {
        tuple<double, double> lyapunov_exponents = wolf_method_henon(get<0>(initial_condition), get<1>(initial_condition), tf, dt, scale);
        lyapunov_sum1 += get<0>(lyapunov_exponents);
        lyapunov_sum2 += get<1>(lyapunov_exponents);
    }

    double lyapunov_exponent1 = lyapunov_sum1;
    double lyapunov_exponent2 = lyapunov_sum2;

    cout << "Lyapunov exponents: " << endl;
    cout << "  Lambda 1 = " << lyapunov_exponent1 << endl;
    cout << "  Lambda 2 = " << lyapunov_exponent2 << endl;

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2);
}




// QR Factorization method attempts: (NOT CURRENTLY WORKING)
Matrix3d jacobian(double x, double y, double z, double sigma, double rho, double beta) {
    Matrix3d J;

    J(0, 0) = -sigma;
    J(0, 1) = sigma;
    J(0, 2) = 0;

    J(1, 0) = rho - z;
    J(1, 1) = -1;
    J(1, 2) = -x;

    J(2, 0) = y;
    J(2, 1) = x;
    J(2, 2) = -beta;

    // Print the jacobian matrix
//    std::cout << endl << "J(0,0): " << J(0,0) << " sigma: " << sigma << std::endl;
//	std::cout << "J(0,1): " << J(0,1) << " sigma: " << sigma << std::endl;
//	std::cout << "J(0,2): " << J(0,2) << std::endl;
//	std::cout << "J(1,0): " << J(1,0) << " rho: " << rho << " z: " << z << std::endl;
//	std::cout << "J(1,1): " << J(1,1) << std::endl;
//	std::cout << "J(1,2): " << J(1,2) << " x: " << x << std::endl;
//	std::cout << "J(2,0): " << J(2,0) << " y: " << y << std::endl;
//	std::cout << "J(2,1): " << J(2,1) << " x: " << x << std::endl;
//	std::cout << "J(2,2): " << J(2,2) << " beta: " << beta << std::endl;

    return J;
}

tuple<double, double, double> qrFactorization_lorenz(double x0, double y0, double z0, double tf, double dt) {
    int n = static_cast<int>((tf - 20.0) / dt);

    double x = x0;
    double y = y0;
    double z = z0;

    cout << "z is: " << z << endl;

    double lyapunov_sum1 = 0;
    double lyapunov_sum2 = 0;
    double lyapunov_sum3 = 0;

    // Initialize Q matrix as the identity matrix
    Matrix3d Q = Matrix3d::Identity();

    // Ignore the first 20 seconds of the simulation
//    for (int i = 0; i < static_cast<int>(20.0 / dt); ++i) {
//        tie(x, y, z) = rk4(x, y, z, dt);
//    }

    for (int i = 0; i < 100; ++i) {
    	tie(x, y, z) = rk4_lorenz(x, y, z, dt);

//    	cout << "Iteration: " << i << endl;

		// Calculate the Jacobian matrix A
		Matrix3d A = jacobian(x, y, z, s, r, b);

//		cout << "Jacobian A: " << endl << A << endl;
//		cout << "Q: " << endl << Q << endl;

		// Orthogonalize and normalize the columns of the Q matrix
		double x1 = Q(0, 0), y1 = Q(1, 0), z1 = Q(2, 0);
		double x2 = Q(0, 1), y2 = Q(1, 1), z2 = Q(2, 1);
		double x3 = Q(0, 2), y3 = Q(1, 2), z3 = Q(2, 2);
		orthogonalizeAndNormalize(x1, y1, z1, x2, y2, z2, x3, y3, z3);
		Q << x1, x2, x3,
			 y1, y2, y3,
			 z1, z2, z3;

		// Compute the product of the current Q matrix and the Jacobian matrix A
		Matrix3d QA = Q * A;
//		cout << "Q: after orthogonalization and normalization" << endl << Q << endl;
//
//		cout << "QA: " << endl << QA << endl;

		// Perform QR factorization on the QA matrix
		HouseholderQR<Matrix3d> qr(QA);
		Matrix3d R = qr.matrixQR().triangularView<Upper>();
//		cout << "R: " << endl << R << endl;
		Q = qr.householderQ();
//		cout << "Q: after qrthing" << endl << Q << endl << endl;

        // Check for orthogonalization
		double error = (Q * Q.transpose() - Matrix3d::Identity()).norm();
		if (error > 1e-6) {
			cout << "Warning: Q is not orthogonal!" << endl;
		}

        // Add the logarithm of the absolute value of the diagonal elements of R to the Lyapunov exponent sums
        lyapunov_sum1 += log(abs(R(0, 0)));
        lyapunov_sum2 += log(abs(R(1, 1)));
        lyapunov_sum3 += log(abs(R(2, 2)));

    }

    // Calculate the average Lyapunov exponents
    double lyapunov_exponent1 = lyapunov_sum1 / (n * dt);
    double lyapunov_exponent2 = lyapunov_sum2 / (n * dt);
    double lyapunov_exponent3 = lyapunov_sum3 / (n * dt);

    return make_tuple(lyapunov_exponent1, lyapunov_exponent2, lyapunov_exponent3);
}



int main() {

	// Standard params
	double tf = 10000.0;
	double dt = 0.01;
	int n_points = 100;

	// Lorenz
	cout << "STARTING LORENZ" << endl;
	double lorenz_scale = 0.00000001;

	// Averaged over many points
	vector<tuple<double, double, double>> lorenz_initial_conditions = generate_even_initial_conditions(1000, -30, 30, -30, 30, 5, 45);
	tuple<double, double, double> lorenz_lyapunov_exponents = multiple_lorenz_results(lorenz_initial_conditions, tf, dt, lorenz_scale);
	// Single point call
	cout << "Single" << endl;
	tuple<double, double, double> result = single_lorenz_results(1.0, 1.0, 1.0, 10000.0, 0.01, 0.00000001);
	// QR factorization (NOT WORKING)
	// tuple<double, double, double> lyapunov_exponents = qrFactorization(x0, y0, z0, tf, dt);


	// ROSSLER
	cout << endl << endl << "ROSSLER" << endl << endl;
	double rossler_scale = 0.00000001;
	// Averaged over many points
	vector<tuple<double, double, double>> rossler_initial_conditions = generate_even_initial_conditions(1000, -20, 20, -20, 20, 0, 30);
	tuple<double, double, double> rossler_lyapunov_exponents = multiple_rossler_results(rossler_initial_conditions, tf, dt, rossler_scale);
	// Single point call
	cout << "Single" << endl;
	tuple<double, double, double> single_rossler_result = single_rossler_results(1.0, 1.0, 1.0, tf, dt, rossler_scale);


	// Rabinovich-Fabrikant
	cout << endl << endl << endl << "RABINOVICH" << endl << endl;
	double rabinovich_scale = 0.00001;
	// Averaged over many points
	vector<tuple<double, double, double>> rabinovich_initial_conditions = generate_even_initial_conditions(1000, -10.0, 10.0, -10.0, 10.0, 0.0, 5.0);
	vector<tuple<double, double>> ab_values = {make_tuple(0.1, 0.98), make_tuple(0.1, 0.5), make_tuple(0.1, 0.2715), make_tuple(-1, -0.1)};

	for (auto& ab : ab_values) {
	    double a = get<0>(ab);
	    double b = get<1>(ab);
	    cout << "a: " << a << " b: " << b << endl;
	    tuple<double, double, double> rabinovich_lyapunov_exponents = multiple_rabinovich_fabrikant_results(rabinovich_initial_conditions, tf, dt, rabinovich_scale, a, b);
	}

	// Single point calls
	cout << "Single" << endl;
	tuple<double, double, double> single_rabinovich_result = single_rabinovich_fabrikant_results(1.0, 1.0, 1.0, tf, dt, 0.1, 0.98, rabinovich_scale);

	// Chua circuit
	cout << endl << endl << "Chua" << endl;
	// Averaged over many points
	double chua_scale = 0.00001;
	vector<tuple<double, double, double>> chua_initial_conditions = generate_even_initial_conditions(1000, -3.0, 3.0, -3.0, 3.0, -5.0, 5.0);
	tuple<double, double, double> chua_lyapunov_exponents = multiple_chua_results(chua_initial_conditions, tf, dt, chua_scale);
	// Single call
	cout << "Single" << endl;
	tuple<double, double, double> single_chua_result = single_chua_results(1.0, 1.0, 1.0, 10000.0, 0.01, 10.0);

	// Henon Map
	cout << endl << endl << "Henon" << endl;
	// Averaged over many points
	double henon_scale = 0.00001;
	vector<tuple<double, double>> henon_initial_conditions = generate_even_initial_conditions_2d(1000, -2.0, 2.0, -2.0, 2.0);
	tuple<double, double> henon_lyapunov_exponents = multiple_henon_results(henon_initial_conditions, tf, dt, chua_scale);
	// Single point call
	cout << "Single" << endl;
	tuple<double, double> single_henon_result = single_henon_results(1.0, 1.0, tf, dt, henon_scale);


	return 0;
}
