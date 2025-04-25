#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <algorithm>
#include <immintrin.h> // For AVX/SSE instructions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// External interface types
using Point2D = std::tuple<double, double>;
using Line2D = std::tuple<Point2D, Point2D>;

// Internal SIMD types
struct Point2DSimd
{
    __m128d vec;

    Point2DSimd() : vec(_mm_setzero_pd()) {}
    Point2DSimd(double x, double y) : vec(_mm_set_pd(y, x)) {}
    Point2DSimd(const Point2D &p) : vec(_mm_set_pd(std::get<1>(p), std::get<0>(p))) {}
    Point2DSimd(__m128d v) : vec(v) {}

    Point2D to_point2d() const
    {
        double arr[2];
        _mm_store_pd(arr, vec);
        return std::make_tuple(arr[0], arr[1]);
    }

    double x() const { return _mm_cvtsd_f64(vec); }
    double y() const { return _mm_cvtsd_f64(_mm_permute_pd(vec, 1)); }
};

struct Line2DSimd
{
    Point2DSimd start;
    Point2DSimd end;

    Line2DSimd(const Point2D &s, const Point2D &e)
        : start(s), end(e) {}
};

// SIMD-optimized utility functions
double distance_simd(const Point2DSimd &p0, const Point2DSimd &p1)
{
    __m128d diff = _mm_sub_pd(p0.vec, p1.vec);
    __m128d squared = _mm_mul_pd(diff, diff);
    __m128d sum = _mm_hadd_pd(squared, squared);
    return std::sqrt(_mm_cvtsd_f64(sum));
}

bool segments_intersect_simd(const Point2DSimd &start0, const Point2DSimd &end0,
                             const Point2DSimd &start1, const Point2DSimd &end1)
{
    // Calculate direction vectors
    __m128d dir0 = _mm_sub_pd(end0.vec, start0.vec);
    __m128d dir1 = _mm_sub_pd(end1.vec, start1.vec);

    // Calculate determinant using SIMD
    __m128d dir0_perm = _mm_permute_pd(dir0, 1);
    __m128d dir1_perm = _mm_permute_pd(dir1, 1);
    __m128d det_vec = _mm_mul_pd(dir0, dir1_perm);
    det_vec = _mm_hsub_pd(det_vec, _mm_mul_pd(dir0_perm, dir1));
    double det = _mm_cvtsd_f64(det_vec);

    if (std::abs(det) <= 0.00001)
    {
        return false;
    }

    // Calculate relative position
    __m128d rel_pos = _mm_sub_pd(start1.vec, start0.vec);
    __m128d rel_pos_perm = _mm_permute_pd(rel_pos, 1);

    // Calculate s using SIMD
    __m128d s_num = _mm_mul_pd(rel_pos, dir1_perm);
    s_num = _mm_hsub_pd(s_num, _mm_mul_pd(rel_pos_perm, dir1));
    double s = _mm_cvtsd_f64(s_num) / det;

    // Calculate t using SIMD
    __m128d t_num = _mm_mul_pd(rel_pos, dir0_perm);
    t_num = _mm_hsub_pd(t_num, _mm_mul_pd(rel_pos_perm, dir0));
    double t = _mm_cvtsd_f64(t_num) / det;

    return (0 <= s && s <= 1) && (0 <= t && t <= 1);
}

Point2DSimd steer_simd(const Point2DSimd &start, const Point2DSimd &end, double radius)
{
    double dist = distance_simd(start, end);
    if (dist < radius)
    {
        return end;
    }

    // Calculate direction
    __m128d diff = _mm_sub_pd(end.vec, start.vec);
    __m128d squared = _mm_mul_pd(diff, diff);
    __m128d sum = _mm_hadd_pd(squared, squared);
    double norm = std::sqrt(_mm_cvtsd_f64(sum));

    // Normalize and scale
    __m128d inv_norm = _mm_set1_pd(1.0 / norm);
    __m128d direction = _mm_mul_pd(diff, inv_norm);
    __m128d scaled = _mm_mul_pd(direction, _mm_set1_pd(radius));
    __m128d result = _mm_add_pd(start.vec, scaled);

    return Point2DSimd(result);
}

// RRT* implementation with SIMD optimization
class Node
{
public:
    Node *parent;
    Point2DSimd pos;
    double cost;
    std::vector<Node *> children;

    Node(Node *parent, const Point2DSimd &position, double cost)
        : parent(parent), pos(position), cost(cost) {}

    void change_cost(double delta_cost)
    {
        cost += delta_cost;
        for (auto child : children)
        {
            child->change_cost(delta_cost);
        }
    }
};

class Map
{
public:
    std::vector<Line2DSimd> obstacles;

    Map(const std::vector<Line2D> &obstacles)
    {
        for (const auto &obs : obstacles)
        {
            this->obstacles.emplace_back(std::get<0>(obs), std::get<1>(obs));
        }
    }

    bool intersects_obstacle(const Point2DSimd &start, const Point2DSimd &end) const
    {
        for (const auto &obstacle : obstacles)
        {
            if (segments_intersect_simd(start, end, obstacle.start, obstacle.end))
            {
                return true;
            }
        }
        return false;
    }
};

class RRTTree
{
public:
    Node *root;

    RRTTree(const Point2DSimd &root_pos)
    {
        root = new Node(nullptr, root_pos, 0);
    }

    ~RRTTree()
    {
        delete_tree(root);
    }

    void delete_tree(Node *node)
    {
        for (auto child : node->children)
        {
            delete_tree(child);
        }
        delete node;
    }

    Node *nearest(const Point2DSimd &point) const
    {
        Node *closest_node = root;
        double min_dist = distance_simd(point, root->pos);

        std::vector<Node *> nodes_to_check = {root};
        while (!nodes_to_check.empty())
        {
            Node *current = nodes_to_check.back();
            nodes_to_check.pop_back();

            double dist = distance_simd(point, current->pos);
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_node = current;
            }

            for (auto child : current->children)
            {
                nodes_to_check.push_back(child);
            }
        }

        return closest_node;
    }

    std::vector<Node *> in_neighborhood(const Point2DSimd &point, double radius) const
    {
        std::vector<Node *> neighbors;
        std::vector<Node *> nodes_to_check = {root};

        while (!nodes_to_check.empty())
        {
            Node *current = nodes_to_check.back();
            nodes_to_check.pop_back();

            if (distance_simd(point, current->pos) < radius)
            {
                neighbors.push_back(current);
            }

            for (auto child : current->children)
            {
                nodes_to_check.push_back(child);
            }
        }

        return neighbors;
    }

    void add_node(Node *parent, Node *node)
    {
        parent->children.push_back(node);
        node->parent = parent;
    }
};

std::vector<Point2D> path_to_root(Node *node)
{
    std::vector<Point2D> path;
    while (node != nullptr)
    {
        path.push_back(node->pos.to_point2d());
        node = node->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Point2D> rrt_star_simd(
    const Point2D &start,
    const Point2D &end,
    const std::vector<Line2D> &obstacles,
    double max_step_size,
    double neighborhood_size,
    int max_iterations)
{
    Point2DSimd start_simd(start);
    Point2DSimd end_simd(end);
    Map map(obstacles);
    RRTTree tree(start_simd);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0, 2);
    std::uniform_real_distribution<> dis_y(0, 1);

    Node *end_node = nullptr;
    int iter_found = -1;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        Point2DSimd random_point(dis_x(gen), dis_y(gen));
        Node *closest_node = tree.nearest(random_point);
        Point2DSimd new_point = steer_simd(closest_node->pos, random_point, max_step_size);

        if (!map.intersects_obstacle(closest_node->pos, new_point))
        {
            auto close_nodes = tree.in_neighborhood(new_point, neighborhood_size);

            Node *min_node = closest_node;
            double min_cost = closest_node->cost + distance_simd(closest_node->pos, new_point);

            for (auto node : close_nodes)
            {
                if (!map.intersects_obstacle(node->pos, new_point))
                {
                    double cost = node->cost + distance_simd(node->pos, new_point);
                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        min_node = node;
                    }
                }
            }

            Node *added_node = new Node(min_node, new_point, min_cost);
            tree.add_node(min_node, added_node);

            // Rewiring
            for (auto node : close_nodes)
            {
                double cost = added_node->cost + distance_simd(added_node->pos, node->pos);
                if (!map.intersects_obstacle(new_point, node->pos) && cost < node->cost)
                {
                    if (node->parent != nullptr)
                    {
                        auto &children = node->parent->children;
                        children.erase(std::remove(children.begin(), children.end(), node), children.end());
                    }
                    node->parent = added_node;
                    node->change_cost(cost - node->cost);
                    added_node->children.push_back(node);
                }
            }

            if (distance_simd(new_point, end_simd) < max_step_size &&
                !map.intersects_obstacle(new_point, end_simd) &&
                end_node == nullptr)
            {
                end_node = new Node(added_node, end_simd, added_node->cost + distance_simd(new_point, end_simd));
                tree.add_node(added_node, end_node);
                iter_found = iter;
            }
        }

        if (end_node != nullptr && iter >= iter_found * 3)
        {
            break;
        }
    }

    if (end_node != nullptr)
    {
        return path_to_root(end_node);
    }
    return {};
}

// Python bindings
PYBIND11_MODULE(rrt_star_simd_full_bind, m)
{
    m.def("rrt_star_simd", &rrt_star_simd, "Run RRT* algorithm with SIMD optimization");
}