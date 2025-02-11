import networkx as nx
import matplotlib.pyplot as plt
import math

def euclidean_distance(node1, node2):
    return round(math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2), 2)


unknown = 266
# Define intersections from the table as (X, Y) coordinates
intersections = {
    (452, 29): "Aquatic Ave. & Beak St.",
    (305, 29): "Aquatic Ave. & Feather St.",
    (129, 29): "Aquatic Ave. & Waddle Way",
    (213, 29): "Aquatic Ave. & Waterfoul Way",
    (284, 393): "Breadcrumb Ave. & The Circle",
    (181, 459): "Breadcrumb Ave. & Waddle Way",
    (305, unknown): "The Circle & Feather St.",
    (273, 307): "The Circle & Waterfoul Way",
    (452, 293): "Dabbler Dr. & Beak St.",
    (350, 324): "Dabbler Dr. & The Circle",
    (585, 293): "Dabbler Dr. & Mallard St.",
    (452, 402): "Drake Dr. & Beak St.",
    (576, 354): "Drake Dr. & Mallard St.",
    (452, 474): "Duckling Dr. & Beak St.",
    (593, 354): "Duckling Dr. & Mallard St.",
    (452, 135): "Migration Ave. & Beak St.",
    (305, 135): "Migration Ave. & Feather St.",
    (585, 135): "Migration Ave. & Mallard St.",
    (29, 135): "Migration Ave. & Quack St.",
    (129, 135): "Migration Ave. & Waddle Way",
    (213, 135): "Migration Ave. & Waterfoul Way",
    (452, 233): "Pondside Ave. & Beak St.",
    (305, 233): "Pondside Ave. & Feather St.",
    (585, 233): "Pondside Ave. & Mallard St.",
    (28, 329): "Pondside Ave. & Quack St.",
    (214, 241): "Pondside Ave. & Waterfoul Way",
    (157, 266): "Pondside Ave. & Waddle Way",
    (452, 465): "Tail Ave. & Beak St.",
    (335, 387): "Tail Ave. & The Circle"
}

levels = {
    ((181, 459), (157, 266)): 2,
    ((28, 329), (157, 266)): 2,
    ((157, 266), (28, 329)): 2,
    ((305, 233), (452, 233)): 2,
    ((452, 233), (305, 233)): 2,
    ((29, 135), (129, 135)): 3,
    ((129, 135), (29, 135)): 3,
    ((129, 135), (213, 135)): 3,
    ((213, 135), (129, 135)): 3,
    ((213, 135), (305, 135)): 3,
    ((305, 135), (213, 135)): 3,
    
}

# Create graph
G = nx.DiGraph()
for (x, y), name in intersections.items():
    G.add_node((x, y), label=name)

# Add edges based on the horizontal and vertical roads
edges = [
    ((181, 459), (28, 329)), ((28, 329), (181, 459)),
    ((181, 459), (284, 393)), ((284, 393), (181, 459)),
    ((28, 329), (157, 266)), ((157, 266), (28, 329)),
    ((29, 135), (129, 135)), ((129, 135), (29, 135)),
    ((28, 329), (29, 135)), ((29, 135), (28, 329)),
    ((29, 135), (129, 29)), ((129, 29), (29, 135)),
    ((157, 266), (214, 241)), ((214, 241), (157, 266)),
    ((129, 135), (213, 135)), ((213, 135), (129, 135)),
    ((129, 29), (213, 29)), ((213, 29), (129, 29)),
    ((213, 29), (305, 29)), ((305, 29), (213, 29)),
    ((305, 29), (452, 29)), ((452, 29), (305, 29)),
    ((585, 135), (585, 233)), ((585, 233), (585, 135)),
    ((585, 233), (585, 293)), ((585, 293), (585, 233)),
    ((452, 474), (452, 465)), ((452, 465), (452, 474)),
    ((452, 465), (452, 402)), ((452, 402), (452, 465)),
    ((452, 402), (452, 293)), ((452, 293), (452, 402)),
    ((452, 293), (452, 233)), ((452, 233), (452, 293)),
    ((452, 233), (452, 135)), ((452, 135), (452, 233)),
    ((452, 135), (452, 29)), ((452, 29), (452, 135)),
    ((213, 135), (305, 135)), ((305, 135), (213, 135)),
    ((305, 135), (452, 135)), ((452, 135), (305, 135)),
    ((452, 135), (585, 135)), ((585, 135), (452, 135)),
    ((585, 135), (452, 29)), ((452, 29), (585, 135)),
    ((214, 241), (305, 233)), ((305, 233), (214, 241)),
    ((305, 233), (452, 233)), ((452, 233), (305, 233)),
    ((452, 233), (585, 233)), ((585, 233), (452, 233)),
    ((395, 29), (305, 135)), ((305, 135), (395, 29)),
    ((305, 135), (305, 233)), ((305, 233), (305, 135)),
    ((305, 233), (305, 266)), ((305, 266), (305, 233)),
    ((335, 387), (452, 474)), ((452, 474), (335, 387)),
    ((335, 387), (452, 465)), ((452, 465), (335, 387)),
    ((157, 266), (129, 135)),
    ((129, 135), (129, 29)),
    ((181, 459), (157, 266)),
    ((213, 135), (214, 241)),
    ((213, 29), (213, 135)),
    ((214, 241), (273, 307)),
    ((273, 307), (305, unknown)),
    ((305, unknown), (350, 324)),
    ((350, 324), (335, 387)),
    ((335, 387), (284, 393)),
    ((284, 393), (273, 307)),
    ((452, 293), (350, 324)),
    ((585, 293), (452, 293)),
    ((585, 293), (593, 354)),
    ((576, 354), (585, 293)),
    ((452, 402), (576, 354)),
    ((593, 354), (452, 474)),
]

for u, v in edges:
    distance = euclidean_distance(u, v)
    level = levels.get((u, v), 1)  # Default level is 1 if not specified
    weight = distance * level  # Weight is distance multiplied by pedestrian level
    G.add_edge(u, v, weight=round(weight, 2))

    G.add_edge(u, v, weight=weight)

# Draw the graph
pos = {node: node for node in G.nodes()}  # Position nodes by their coordinates
labels = nx.get_edge_attributes(G, 'weight')  # Get edge weights


# VISUALIZE GRAPH
# plt.figure(figsize=(8, 8))
# nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightblue", font_size=8, edge_color="gray")
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

# plt.show()

def find_shortest_path(start, end):
    """Finds and visualizes the shortest path using Dijkstra's algorithm."""
    # Compute shortest path
    shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight', method='dijkstra')
    print("Shortest Path:", shortest_path)
    # Get edge list for shortest path
    path_edges = list(zip(shortest_path, shortest_path[1:]))

    # Draw graph
    plt.figure(figsize=(10, 6))
    pos = {node: node for node in G.nodes()}  # Use coordinates as positions
    labels = {node: node for node in intersections}  # Use intersection names

    # Draw all edges
    nx.draw(G, pos, node_size=300, node_color="lightgray", with_labels=False, edge_color="gray", width=1)
    
    # Draw shortest path edges
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

    # Draw nodes with names
    for node, (x, y) in pos.items():
        plt.text(x, y, labels.get(node, ""), fontsize=8, ha="right", va="bottom", color="black")

    # Show plot
    plt.title(f"Shortest Path from {intersections[start]} to {intersections[end]}")
    plt.show()

# Example usage
start = (305, 29)  # Example start node
end = (157, 266)    # Example end node
find_shortest_path(start, end)
