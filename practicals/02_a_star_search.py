"""
Practical 02: Implement the A* search algorithm.

A* evaluates a node using f(n) = g(n) + h(n), where g is the cost
from the start and h is the estimated cost to the goal.
"""

import heapq
from itertools import count


def a_star_search(graph, heuristic, start, goal):
    """
    Find a minimum-cost path.

    graph: {node: [(neighbour, edge_cost), ...]}
    heuristic: {node: estimated_cost_to_goal}
    Returns (path, total_cost), or ([], float("inf")) if unreachable.
    """
    if start not in graph or goal not in graph:
        raise ValueError("Start and goal must be graph nodes.")

    tie_breaker = count()
    frontier = [(heuristic.get(start, 0), next(tie_breaker), 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, _, current_cost, current = heapq.heappop(frontier)

        # Ignore outdated queue entries.
        if current_cost != cost_so_far.get(current):
            continue

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path)), current_cost

        for neighbour, edge_cost in graph.get(current, []):
            if edge_cost < 0:
                raise ValueError("A* requires non-negative edge costs.")

            new_cost = current_cost + edge_cost
            if new_cost < cost_so_far.get(neighbour, float("inf")):
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current
                priority = new_cost + heuristic.get(neighbour, 0)
                heapq.heappush(
                    frontier,
                    (priority, next(tie_breaker), new_cost, neighbour),
                )

    return [], float("inf")


def main():
    # Weighted example graph. Heuristic values estimate distance to G.
    graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("A", 1), ("D", 2), ("E", 5)],
        "C": [("A", 4), ("F", 3)],
        "D": [("B", 2), ("G", 5)],
        "E": [("B", 5), ("G", 2)],
        "F": [("C", 3), ("G", 2)],
        "G": [],
    }

    heuristic = {"A": 5, "B": 4, "C": 3, "D": 3, "E": 1, "F": 2, "G": 0}
    start, goal = "A", "G"

    path, cost = a_star_search(graph, heuristic, start, goal)

    print("A* Search Algorithm")
    print("-------------------")
    print(f"Start: {start} | Goal: {goal}")
    print("Heuristic values:", heuristic)
    if path:
        print("Optimal path:", " -> ".join(path))
        print("Path cost:", cost)
    else:
        print("No path exists.")


if __name__ == "__main__":
    main()
