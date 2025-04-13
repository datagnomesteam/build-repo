import json
from collections import defaultdict, deque

def load_mapping(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    graph = defaultdict(list)
    all_nodes = set()

    for op in data:
        for entry in op['edits']:
            to_value = entry['to'].strip().lower()
            for from_value in entry['from']:
                from_value = from_value.strip().lower()
                if from_value != to_value:  # skip redundant self-maps
                    graph[from_value].append(to_value)
                    all_nodes.add(from_value)
                    all_nodes.add(to_value)

    return graph, all_nodes

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


def find_longest_path(graph, all_nodes):
    memo = {}

    def dfs(node, visiting):
        if node in memo:
            return memo[node]
        if node in visiting:
            # Cycle detected
            return 0

        visiting.add(node)
        max_len = 0

        for neighbor in graph.get(node, []):
            max_len = max(max_len, dfs(neighbor, visiting))

        visiting.remove(node)
        memo[node] = 1 + max_len
        return memo[node]

    longest = 0
    for node in all_nodes:
        longest = max(longest, dfs(node, set()))

    return longest

# Example usage
if __name__ == "__main__":
    graph, all_nodes = load_mapping("openrefine.json")
    cycles = find_all_cycles(graph)
    if cycles:
        print(f"🔁 {len(cycles)} loop(s) detected:")
        for idx, cycle in enumerate(cycles, 1):
            print(f"{idx}. {' → '.join(cycle)} → {cycle[0]}")
    else:
        print("✅ No cycles found.")
    
    max_hops = find_longest_path(graph, all_nodes)


    print(f"Maximum hops required to reach the final value in the cycle: {max_hops}")
