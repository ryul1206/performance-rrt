#include <vector>
#include <cmath>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Point2D = std::tuple<double, double>;
using Line2D = std::tuple<Point2D, Point2D>;

// Implementation
double distance(const Point2D &p0, const Point2D &p1)
{
    double dx = std::get<0>(p0) - std::get<0>(p1);
    double dy = std::get<1>(p0) - std::get<1>(p1);
    return std::sqrt(dx * dx + dy * dy);
}

bool segments_intersect(const Point2D &start0, const Point2D &end0,
                        const Point2D &start1, const Point2D &end1)
{
    double dir0_x = std::get<0>(end0) - std::get<0>(start0);
    double dir0_y = std::get<1>(end0) - std::get<1>(start0);
    double dir1_x = std::get<0>(end1) - std::get<0>(start1);
    double dir1_y = std::get<1>(end1) - std::get<1>(start1);

    double det = dir0_x * dir1_y - dir0_y * dir1_x;
    if (std::abs(det) <= 0.00001)
    {
        return false;
    }

    double dx = std::get<0>(start1) - std::get<0>(start0);
    double dy = std::get<1>(start1) - std::get<1>(start0);

    double s = (dx * dir1_y - dy * dir1_x) / det;
    double t = (dx * dir0_y - dy * dir0_x) / det;

    return (0 <= s && s <= 1) && (0 <= t && t <= 1);
}

Point2D steer(const Point2D &start, const Point2D &end, double radius)
{
    double dist = distance(start, end);
    if (dist < radius)
    {
        return end;
    }

    double diff_x = std::get<0>(end) - std::get<0>(start);
    double diff_y = std::get<1>(end) - std::get<1>(start);
    double norm = std::sqrt(diff_x * diff_x + diff_y * diff_y);

    double direction_x = diff_x / norm;
    double direction_y = diff_y / norm;

    return std::make_tuple(
        direction_x * radius + std::get<0>(start),
        direction_y * radius + std::get<1>(start));
}

// Python bindings
PYBIND11_MODULE(rrt_star_cpp_fn_bind, m)
{
    m.def("distance", &distance, "Calculate distance between two points");
    m.def("segments_intersect", &segments_intersect, "Check if two segments intersect");
    m.def("steer", &steer, "Find the point in a disc around start that is closest to end");
}