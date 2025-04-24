#include <vector>
#include <cmath>
#include <tuple>
#include <random>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Point2D = std::tuple<double, double>;
using Line2D = std::tuple<Point2D, Point2D>;

// Utility functions
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

// RRT* implementation
class Node
{
public:
    Node *parent;
    Point2D pos;
    double cost;
    std::vector<Node *> children;

    Node(Node *parent, const Point2D &position, double cost)
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
    std::vector<Line2D> obstacles;

    Map(const std::vector<Line2D> &obstacles) : obstacles(obstacles) {}

    bool intersects_obstacle(const Point2D &start, const Point2D &end) const
    {
        for (const auto &obstacle : obstacles)
        {
            if (segments_intersect(start, end, std::get<0>(obstacle), std::get<1>(obstacle)))
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

    RRTTree(const Point2D &root_pos)
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

    Node *nearest(const Point2D &point) const
    {
        Node *closest_node = root;
        double min_dist = distance(point, root->pos);

        std::vector<Node *> nodes_to_check = {root};
        while (!nodes_to_check.empty())
        {
            Node *current = nodes_to_check.back();
            nodes_to_check.pop_back();

            double dist = distance(point, current->pos);
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

    std::vector<Node *> in_neighborhood(const Point2D &point, double radius) const
    {
        std::vector<Node *> neighbors;
        std::vector<Node *> nodes_to_check = {root};

        while (!nodes_to_check.empty())
        {
            Node *current = nodes_to_check.back();
            nodes_to_check.pop_back();

            if (distance(point, current->pos) < radius)
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
        path.push_back(node->pos);
        node = node->parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Point2D> rrt_star(
    const Point2D &start,
    const Point2D &end,
    const std::vector<Line2D> &obstacles,
    double max_step_size,
    double neighborhood_size,
    int max_iterations)
{
    Map map(obstacles);
    RRTTree tree(start);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0, 2);
    std::uniform_real_distribution<> dis_y(0, 1);

    Node *end_node = nullptr;
    int iter_found = -1;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        Point2D random_point = std::make_tuple(dis_x(gen), dis_y(gen));
        Node *closest_node = tree.nearest(random_point);
        Point2D new_point = steer(closest_node->pos, random_point, max_step_size);

        if (!map.intersects_obstacle(closest_node->pos, new_point))
        {
            auto close_nodes = tree.in_neighborhood(new_point, neighborhood_size);

            Node *min_node = closest_node;
            double min_cost = closest_node->cost + distance(closest_node->pos, new_point);

            for (auto node : close_nodes)
            {
                if (!map.intersects_obstacle(node->pos, new_point))
                {
                    double cost = node->cost + distance(node->pos, new_point);
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
                double cost = added_node->cost + distance(added_node->pos, node->pos);
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

            if (distance(new_point, end) < max_step_size &&
                !map.intersects_obstacle(new_point, end) &&
                end_node == nullptr)
            {
                end_node = new Node(added_node, end, added_node->cost + distance(new_point, end));
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
PYBIND11_MODULE(rrt_star_cpp_full_bind, m)
{
    m.def("rrt_star", &rrt_star, "Run RRT* algorithm");
}