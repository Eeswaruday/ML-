def dfs(graph, start_node):
    visited = set()
    stack = [start_node]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            result.append(node)  # Process the node
            visited.add(node)
            # Ensure correct order of neighbors for expected DFS output
            stack.extend(reversed(graph[node]))

    print(" ".join(result))  # Print final traversal result


# Define the graph
graph = {
    "1": ["4", "2"],
    "2": ["8", "7", "5", "3", "1"],
    "3": ["10", "9", "4", "2"],
    "4": ["3", "1"],
    "5": ["6", "8", "7", "2"],
    "6": [],
    "7": ["8", "5", "2"],
    "8": ["7", "5", "2"],
    "9": [],
    "10": []
}

# Run DFS from "1"
print("DFS Traversal:")
dfs(graph, "1")
