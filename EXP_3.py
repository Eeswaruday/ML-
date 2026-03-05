from queue import Queue

def bfs(graph, start_node):
    visited = set()
    queue = Queue()
    queue.put(start_node)

    while not queue.empty():
        node = queue.get()
        if node not in visited:
            print(node, end=' ')  # Process the node
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.put(neighbor)

# Define the graph
graph = {
    "1": ["4", "2"],
    "2": ["1", "3", "5", "7", "8"],
    "3": ["2", "4", "9", "10"],
    "4": ["1", "3"],
    "5": ["2", "6", "7", "8"],
    "6": [],
    "7": ["2", "5", "8"],
    "8": ["2", "5", "7"],
    "9": [],
    "10": []
}

# Run BFS from "1"
print("BFS Traversal:")
bfs(graph, "1")
