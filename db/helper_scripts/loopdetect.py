import json
from collections import defaultdict, deque

def load_mapping(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    graph = defaultdict(list)

    for op in data:
        for entry in op['edits']:
            to_value = entry['to']
            for from_value in entry['from']:
                graph[from_value.lower()].append(to_value.lower())  # case-insensitive

    return graph

def find_all_cycles(graph):
    visited = set()
    all_cycles = set()

    def dfs(node, path, stack):
        visited.add(node)
        stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor in stack:
                # Found a cycle — slice the path to get the cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                
                # Exclude self-loops (where the cycle starts and ends at the same node)
                # Exclude redundant cycles (where all elements are the same)
                if len(cycle) > 1 and cycle[0] != cycle[-1] and len(set(cycle)) > 1:
                    normalized_cycle = tuple(sorted(cycle))  # Sort to prevent duplicates
                    all_cycles.add(normalized_cycle)
            elif neighbor not in visited:
                dfs(neighbor, path, stack)

        stack.remove(node)
        path.pop()

    for node in graph:
        if node not in visited:
            dfs(node, [], set())

    return [list(cycle) for cycle in all_cycles]

def find_longest_path(graph):
    visited = set()
    max_hops = 0

    def dfs(node, path, stack):
        nonlocal max_hops
        visited.add(node)
        stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor in stack:
                # Found a cycle — slice the path to get the cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:]
                
                # Calculate the number of hops in the cycle (distance from start to finish)
                hop_count = len(cycle)
                
                # Update max_hops if this is the largest path length seen
                if hop_count > max_hops:
                    max_hops = hop_count
            elif neighbor not in visited:
                dfs(neighbor, path, stack)

        stack.remove(node)
        path.pop()

    for node in graph:
        if node not in visited:
            dfs(node, [], set())

    return max_hops

# Example usage
if __name__ == "__main__":
    graph = load_mapping("openrefine.json")
    cycles = find_all_cycles(graph)
    if cycles:
        print(f"🔁 {len(cycles)} loop(s) detected:")
        for idx, cycle in enumerate(cycles, 1):
            print(f"{idx}. {' → '.join(cycle)} → {cycle[0]}")
    else:
        print("✅ No cycles found.")
    
    max_hops = find_longest_path(graph)

    if max_hops:
        print(f"Maximum hops required to reach the final value in the cycle: {max_hops}")
    else:
        print("✅ No cycles found.")
