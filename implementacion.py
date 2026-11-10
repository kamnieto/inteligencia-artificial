from collections import deque
import heapq
import time

# DEFINICIÓN DEL ESPACIO DE ESTADOS (CampusGuard)

GRAPH = {
    "Alto": [
        ("Medio", "Tutoria", 2),
        ("Medio_B", "Bienestar", 1),
        ("Bajo", "AjusteCarga", 4)
    ],

    "Medio": [
        ("Bajo", "Tutoria", 2),
        ("Bajo", "Bienestar", 1)
    ],

    "Medio_B": [
        ("Bajo", "Tutoria", 2)
    ],

    "Bajo": []
}

INITIAL_STATE = "Alto"
GOAL_STATE = "Bajo"

# HEURÍSTICA PARA A*

HEURISTIC = {
    "Alto": 3,
    "Medio": 2,
    "Medio_B": 1,
    "Bajo": 0
}

# BFS - BREADTH FIRST SEARCH

def bfs(start, goal):

    start_time = time.perf_counter()

    frontier = deque()
    frontier.append((start, [], 0))

    visited = set()

    expanded_nodes = 0

    while frontier:

        state, path, cost = frontier.popleft()

        if state in visited:
            continue

        visited.add(state)
        expanded_nodes += 1

        if state == goal:

            execution_time = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "algorithm": "BFS",
                "path": path,
                "cost": cost,
                "expanded": expanded_nodes,
                "time_ms": execution_time
            }

        for next_state, action, action_cost in GRAPH[state]:

            frontier.append(
                (
                    next_state,
                    path + [(action, next_state)],
                    cost + action_cost
                )
            )

    return None

# DFS - DEPTH FIRST SEARCH

def dfs(start, goal):

    start_time = time.perf_counter()

    frontier = [(start, [], 0)]

    visited = set()

    expanded_nodes = 0

    while frontier:

        state, path, cost = frontier.pop()

        if state in visited:
            continue

        visited.add(state)

        expanded_nodes += 1

        if state == goal:

            execution_time = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "algorithm": "DFS",
                "path": path,
                "cost": cost,
                "expanded": expanded_nodes,
                "time_ms": execution_time
            }

        successors = GRAPH[state]

        for next_state, action, action_cost in reversed(successors):

            frontier.append(
                (
                    next_state,
                    path + [(action, next_state)],
                    cost + action_cost
                )
            )

    return None


# A* SEARCH

def astar(start, goal):

    start_time = time.perf_counter()

    frontier = []

    heapq.heappush(
        frontier,
        (
            HEURISTIC[start],
            0,
            start,
            []
        )
    )

    visited = set()

    expanded_nodes = 0

    while frontier:

        f, g, state, path = heapq.heappop(frontier)

        if state in visited:
            continue

        visited.add(state)
        expanded_nodes += 1

        if state == goal:

            execution_time = (
                time.perf_counter() - start_time
            ) * 1000

            return {
                "algorithm": "A*",
                "path": path,
                "cost": g,
                "expanded": expanded_nodes,
                "time_ms": execution_time
            }

        for next_state, action, action_cost in GRAPH[state]:

            new_g = g + action_cost
            new_h = HEURISTIC[next_state]
            new_f = new_g + new_h

            heapq.heappush(
                frontier,
                (
                    new_f,
                    new_g,
                    next_state,
                    path + [(action, next_state)]
                )
            )

    return None

# IMPRESIÓN DE RESULTADOS

def print_solution(result):

    print("\n" + "=" * 60)
    print(result["algorithm"])
    print("=" * 60)

    print("\nCamino encontrado:")

    for action, state in result["path"]:
        print(f"{action} -> {state}")

    print("\nCosto total:", result["cost"])
    print("Nodos expandidos:", result["expanded"])
    print(
        f"Tiempo CPU: {result['time_ms']:.4f} ms"
    )

# BENCHMARK

def benchmark():

    bfs_result = bfs(
        INITIAL_STATE,
        GOAL_STATE
    )

    dfs_result = dfs(
        INITIAL_STATE,
        GOAL_STATE
    )

    astar_result = astar(
        INITIAL_STATE,
        GOAL_STATE
    )

    print_solution(bfs_result)
    print_solution(dfs_result)
    print_solution(astar_result)

    print("\n")
    print("=" * 70)
    print("TABLA BENCHMARK")
    print("=" * 70)

    print(
        f"{'Algoritmo':<10}"
        f"{'Nodos':<12}"
        f"{'Tiempo(ms)':<15}"
        f"{'Costo':<10}"
    )

    for r in [
        bfs_result,
        dfs_result,
        astar_result
    ]:

        print(
            f"{r['algorithm']:<10}"
            f"{r['expanded']:<12}"
            f"{r['time_ms']:<15.4f}"
            f"{r['cost']:<10}"
        )

# MAIN

if __name__ == "__main__":
    benchmark()