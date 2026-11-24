"""
Practical 01: Implement Breadth First Search (BFS).

BFS explores a graph level by level using a queue. It finds a shortest
path in an unweighted graph when one exists.
"""

from collections import deque


def breadth_first_search(graph, start, goal):
    """Return (path, visit_order) using BFS; path is [] if goal is unreachable."""
    if start not in graph:
        raise ValueError(f"Start node {start!r} is not in the graph.")
    if goal not in graph:
        raise ValueError(f"Goal node {goal!r} is not in the graph.")

    queue = deque([start])
    parent = {start: None}
    visit_order = []

    while queue:
        current = queue.popleft()
        visit_order.append(current)

        if current == goal:
            break

        for neighbour in graph[current]:
            if neighbour not in parent:
                parent[neighbour] = current
                queue.append(neighbour)

    if goal not in parent:
        return [], visit_order

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]

    return list(reversed(path)), visit_order


def main():
    # Undirected example graph represented as an adjacency list.
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "G"],
        "F": ["C", "G"],
        "G": ["E", "F"],
    }

    start, goal = "A", "G"
    path, order = breadth_first_search(graph, start, goal)

    print("Breadth First Search")
    print("--------------------")
    print("Graph:", graph)
    print("Start node:", start)
    print("Goal node:", goal)
    print("Traversal order:", " -> ".join(order))
    if path:
        print("Shortest path:", " -> ".join(path))
        print("Number of edges:", len(path) - 1)
    else:
        print("No path exists.")


if __name__ == "__main__":
    main()
