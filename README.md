Practical 1

Aim:
Breadth First Search & Iterative Depth First Search 
Implement the Breadth First Search algorithm to solve a given problem. 
Implement the Iterative Depth First Search algorithm to solve the same problem. 
Compare the performance and efficiency of both algorithms.

Algorithm:
1.Define Data:
Set up city connections with distances and city coordinates.
2.Create Graph:
Initialize a NetworkX graph and add edges with distances.
3.BFS Levels:
Implement Breadth-First Search (BFS) to calculate node levels.
Use these levels to color nodes based on their distance from the start city.
4.Folium Map:
Create a Folium map centered on Mumbai.
Add city markers with colors representing BFS levels.
Add lines connecting cities based on the graph.
5.BFS Search:
Implement BFS to find the shortest path from Mumbai to Belgaum.
6.Plot and Highlight:
Plot the graph using NetworkX.
Highlight the shortest path found by BFS in red.
7.Save and Display:
Save the Folium map as an HTML file.
Display the NetworkX graph with highlighted BFS path.

Code: BFS
import folium
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Aurangabad': 338, 'Nagpur': 821, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Nashik': 211, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Mumbai': 821, 'Aurangabad': 280, 'Jabalpur': 322},
    'Nashik': {'Mumbai': 167, 'Pune': 211, 'Aurangabad': 199, 'Daman': 132},
    'Aurangabad': {'Mumbai': 338, 'Pune': 235, 'Nagpur': 280, 'Nashik': 199, 'Jalna': 98},
    'Goa': {'Mumbai': 467, 'Belgaum': 123},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Jabalpur': {'Nagpur': 322},
    'Daman': {'Nashik': 132},
    'Jalna': {'Aurangabad': 98},
    'Kolhapur': {'Satara': 130},
    'Belgaum': {'Goa': 123}
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Jabalpur': (23.1762, 79.9559),
    'Daman': (20.3974, 72.8311),
    'Jalna': (19.8403, 75.8878),
    'Kolhapur': (16.7054, 74.2198),
    'Belgaum': (15.8497, 74.4977)
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# BFS Levels function
def bfs_levels(start):
    levels = {start: 0}
    queue = [start]
    while queue:
        node = queue.pop(0)
        current_level = levels[node]
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)
    return levels

levels = bfs_levels('Mumbai')

unique_levels = list(set(levels.values()))
color_map = plt.get_cmap('viridis')
norm = plt.Normalize(min(unique_levels), max(unique_levels))
color_dict = {level: color_map(norm(level)) for level in unique_levels}

node_colors = [color_dict[levels[city]] for city in G.nodes()]

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

for city, (lat, lon) in city_coordinates.items():
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill=True,
        fill_color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill_opacity=0.7,
        tooltip=city
    ).add_to(m)

for u, v in G.edges():
    folium.PolyLine(
        locations=[city_coordinates[u], city_coordinates[v]],
        color='gray',
        weight=2,
        opacity=0.5
    ).add_to(m)

m.save('maharashtra_graph_map.html')

# BFS function
def bfs(graph, start, goal):
    visited = set()
    queue = [start]
    path = {start: [start]}

    while queue:
        node = queue.pop(0)
        if node == goal:
            return path[node]

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path[neighbor] = path[node] + [neighbor]
    
    return None

# Execute BFS
start = 'Mumbai'
goal = 'Belgaum'
result = bfs(G, start, goal)
print("Shortest path from Mumbai to Belgaum using BFS:", result)

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')

# Highlight the shortest path found by BFS
if result:
    path_edges = list(zip(result, result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

plt.show()

Algorithm:
1.Define Data:
Set city connections and coordinates.
2.Create Graph:
Initialize NetworkX graph and add edges with distances.
3.DFS Levels:
Calculate node levels using DFS.
Assign colors based on levels.
4.Folium Map:
Create a map with city markers and connections.
5.Iterative DFS:
Implement DFS to find a path between two cities.
6.Execute and Print:
Find and print the DFS path from Mumbai to Belgaum.
7.Graph Visualization:
Plot graph with NetworkX.
Highlight DFS path in blue.
8.Save and Display:
Save Folium map as HTML.
Display NetworkX graph.

Code: IDFS
import folium
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Aurangabad': 338, 'Nagpur': 821, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Nashik': 211, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Mumbai': 821, 'Aurangabad': 280, 'Jabalpur': 322},
    'Nashik': {'Mumbai': 167, 'Pune': 211, 'Aurangabad': 199, 'Daman': 132},
    'Aurangabad': {'Mumbai': 338, 'Pune': 235, 'Nagpur': 280, 'Nashik': 199, 'Jalna': 98},
    'Goa': {'Mumbai': 467, 'Belgaum': 123},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Jabalpur': {'Nagpur': 322},
    'Daman': {'Nashik': 132},
    'Jalna': {'Aurangabad': 98},
    'Kolhapur': {'Satara': 130},
    'Belgaum': {'Goa': 123}
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Jabalpur': (23.1762, 79.9559),
    'Daman': (20.3974, 72.8311),
    'Jalna': (19.8403, 75.8878),
    'Kolhapur': (16.7054, 74.2198),
    'Belgaum': (15.8497, 74.4977)
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# DFS Levels function
def dfs_levels(start):
    levels = {start: 0}
    stack = [start]
    while stack:
        node = stack.pop()
        current_level = levels[node]
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                stack.append(neighbor)
    return levels

levels = dfs_levels('Mumbai')

unique_levels = list(set(levels.values()))
color_map = plt.get_cmap('viridis')
norm = plt.Normalize(min(unique_levels), max(unique_levels))
color_dict = {level: color_map(norm(level)) for level in unique_levels}

node_colors = [color_dict[levels[city]] for city in G.nodes()]

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

for city, (lat, lon) in city_coordinates.items():
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill=True,
        fill_color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill_opacity=0.7,
        tooltip=city
    ).add_to(m)

for u, v in G.edges():
    folium.PolyLine(
        locations=[city_coordinates[u], city_coordinates[v]],
        color='gray',
        weight=2,
        opacity=0.5
    ).add_to(m)

m.save('maharashtra_graph_map.html')

# Iterative DFS function
def iterative_dfs(graph, start, goal):
    stack = [start]
    visited = set()
    path = {start: [start]}

    while stack:
        node = stack.pop()
        if node == goal:
            return path[node]

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)
                    if neighbor not in path:
                        path[neighbor] = path[node] + [neighbor]

    return None

# Execute DFS
start = 'Mumbai'
goal = 'Belgaum'
result = iterative_dfs(G, start, goal)
print("DFS path from Mumbai to Belgaum:", result)

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')

# Highlight the path found by DFS
if result:
    path_edges = list(zip(result, result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2)

plt.show()

Algorithm:
1.Define Data:
Set up city connections and coordinates.
2.Create Graph:
Initialize NetworkX graph.
Add edges with distances.
3.BFS Levels:
Compute node levels using BFS.
Color nodes based on levels.
4.Folium Map:
Create a map centered on Mumbai.
Add city markers and connections.
5.Algorithms:
Implement BFS and Iterative DFS.
Find paths between cities.
6.Visualization:
Plot the graph with NetworkX.
Highlight BFS and DFS paths.
7.Save Results:
Save Folium map as HTML.
Display NetworkX graph.

Code: Compare
import folium
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Aurangabad': 338, 'Nagpur': 821, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Nashik': 211, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Mumbai': 821, 'Aurangabad': 280, 'Jabalpur': 322},
    'Nashik': {'Mumbai': 167, 'Pune': 211, 'Aurangabad': 199, 'Daman': 132},
    'Aurangabad': {'Mumbai': 338, 'Pune': 235, 'Nagpur': 280, 'Nashik': 199, 'Jalna': 98},
    'Goa': {'Mumbai': 467, 'Belgaum': 123},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Jabalpur': {'Nagpur': 322},
    'Daman': {'Nashik': 132},
    'Jalna': {'Aurangabad': 98},
    'Kolhapur': {'Satara': 130},
    'Belgaum': {'Goa': 123}
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Jabalpur': (23.1762, 79.9559),
    'Daman': (20.3974, 72.8311),
    'Jalna': (19.8403, 75.8878),
    'Kolhapur': (16.7054, 74.2198),
    'Belgaum': (15.8497, 74.4977)
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# BFS Levels function
def bfs_levels(start):
    levels = {start: 0}
    queue = [start]
    while queue:
        node = queue.pop(0)
        current_level = levels[node]
        for neighbor in G.neighbors(node):
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)
    return levels

levels = bfs_levels('Mumbai')

unique_levels = list(set(levels.values()))
color_map = plt.get_cmap('viridis')
norm = plt.Normalize(min(unique_levels), max(unique_levels))
color_dict = {level: color_map(norm(level)) for level in unique_levels}

node_colors = [color_dict[levels[city]] for city in G.nodes()]

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

for city, (lat, lon) in city_coordinates.items():
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill=True,
        fill_color=mcolors.to_hex(node_colors[list(city_coordinates.keys()).index(city)]),
        fill_opacity=0.7,
        tooltip=city
    ).add_to(m)

for u, v in G.edges():
    folium.PolyLine(
        locations=[city_coordinates[u], city_coordinates[v]],
        color='gray',
        weight=2,
        opacity=0.5
    ).add_to(m)

m.save('maharashtra_graph_map.html')

# BFS function
def bfs(graph, start, goal):
    visited = set()
    queue = [start]
    path = {start: [start]}

    while queue:
        node = queue.pop(0)
        if node == goal:
            return path[node]

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path[neighbor] = path[node] + [neighbor]
    
    return None

# Iterative DFS function
def iterative_dfs(graph, start, goal):
    stack = [start]
    visited = set()
    path = {start: [start]}

    while stack:
        node = stack.pop()
        if node == goal:
            return path[node]

        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    stack.append(neighbor)
                    if neighbor not in path:
                        path[neighbor] = path[node] + [neighbor]

    return None

# Execute BFS
start = 'Mumbai'
goal = 'Belgaum'
result = bfs(G, start, goal)
print("Shortest path from Mumbai to Belgaum using BFS:", result)

# Execute DFS
dfs_result = iterative_dfs(G, start, goal)
print("DFS path from Mumbai to Belgaum:", dfs_result)

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')

# Highlight the shortest path found by BFS
if result:
    path_edges = list(zip(result, result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

# Highlight the shortest path found by DFS
if dfs_result:
    dfs_path_edges = list(zip(dfs_result, dfs_result[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=dfs_path_edges, edge_color='blue', width=2, style='dashed')

plt.show()

 
Practical 2

Aim:
A* Search and Recursive Best-First Search 
Implement the A* Search algorithm for solving a pathfinding problem. 
Implement the Recursive Best-First Search algorithm for the same problem. 
Compare the performance and effectiveness of both algorithms.

Algorithm:
1.Initialize Priority Queue:
Start with a queue containing the start node with an initial score of 0 (cost + heuristic).
2.G-Score Tracking:
Maintain g_scores to keep track of the shortest distance from the start to any node.
3.Explore the Best Path:
At each step, explore the node with the lowest f-score (g_score + heuristic).
4.Neighbor Exploration:
For each neighboring node, calculate its tentative g-score (current node g-score + edge weight).
5.Path Update:
If a better path to a neighbor is found, update its g-score and f-score and push the neighbor to the queue.
6.Goal Check:
If the goal node is reached, return the path.
7.Backtrack on Failure:
If the queue is empty, the goal is unreachable.

Code: A* search
import folium
import networkx as nx
import heapq
import time
import matplotlib.pyplot as plt

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Aurangabad': 280},
    'Nashik': {'Mumbai': 167, 'Aurangabad': 199},
    'Aurangabad': {'Pune': 235, 'Nagpur': 280, 'Nashik': 199},
    'Goa': {'Mumbai': 467, 'Kolhapur': 221},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Kolhapur': {'Satara': 130, 'Goa': 221},
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Kolhapur': (16.7054, 74.2198),
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# Heuristic function (straight-line distance to the goal)
def heuristic(node, goal):
    x1, y1 = city_coordinates[node]
    x2, y2 = city_coordinates[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# A* Search function
def a_star_search(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    visited = set()
    g_scores = {start: 0}

    while queue:
        _, current, path = heapq.heappop(queue)
        if current == goal:
            return path
        
        visited.add(current)
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_scores[current] + graph[current][neighbor]['weight']
            if neighbor not in visited or tentative_g_score < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(queue, (f_score, neighbor, path + [neighbor]))
    
    return None

# Execute A* Search
start = 'Mumbai'
goal = 'Kolhapur'

start_time = time.time()
a_star_result = a_star_search(G, start, goal)
a_star_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using A* Search:", a_star_result)
print("A* Search Time:", a_star_time, "seconds")

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

# Add city markers
for city, (lat, lon) in city_coordinates.items():
    folium.Marker(
        location=[lat, lon],
        popup=city,
    ).add_to(m)

# Add paths
if a_star_result:
    for i in range(len(a_star_result) - 1):
        folium.PolyLine(
            locations=[city_coordinates[a_star_result[i]], city_coordinates[a_star_result[i+1]]],
            color='blue',
            weight=2.5,
            opacity=0.7
        ).add_to(m)

# Save the map
m.save('maharashtra_graph_map.html')

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')
plt.show()

Algorithm:
1.Graph Representation:
Cities as nodes and distances as weighted edges.
2.Heuristic:
Use Euclidean distance to estimate the distance to the goal.
3.Recursive Search:
Recursively explore the best path using a cost limit (f_limit).
4.Successor Sorting:
Sort neighbors by estimated total cost (path cost + heuristic).
5.Path Update:
Update and return the path when the goal is reached.
6.Termination:
Backtrack if the path exceeds the cost limit.
7.Output:
Return the shortest path or failure if the goal is unreachable.

Code: RBFS
import folium
import networkx as nx
import time
import matplotlib.pyplot as plt

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Aurangabad': 280},
    'Nashik': {'Mumbai': 167, 'Aurangabad': 199},
    'Aurangabad': {'Pune': 235, 'Nagpur': 280, 'Nashik': 199},
    'Goa': {'Mumbai': 467, 'Kolhapur': 221},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Kolhapur': {'Satara': 130, 'Goa': 221},
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Kolhapur': (16.7054, 74.2198),
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# Heuristic function (straight-line distance to the goal)
def heuristic(node, goal):
    x1, y1 = city_coordinates[node]
    x2, y2 = city_coordinates[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Recursive Best-First Search function
def rbfs(graph, current, goal, f_limit, path, g_scores):
    if current == goal:
        return path, 0

    successors = []
    for neighbor in graph.neighbors(current):
        if neighbor not in path:
            g_score = g_scores[current] + graph[current][neighbor]['weight']
            f_score = g_score + heuristic(neighbor, goal)
            successors.append((neighbor, f_score))
            g_scores[neighbor] = g_score
    
    if not successors:
        return None, float('inf')
    
    successors.sort(key=lambda x: x[1])

    while successors:
        best, f_best = successors[0]
        if f_best > f_limit:
            return None, f_best
        
        alternative = successors[1][1] if len(successors) > 1 else float('inf')
        result, f_best_new = rbfs(graph, best, goal, min(f_limit, alternative), path + [best], g_scores)
        if result:
            return result, f_best_new
        
        successors[0] = (best, f_best_new)
        successors.sort(key=lambda x: x[1])
    
    return None, float('inf')

# Helper function for RBFS
def recursive_best_first_search(graph, start, goal):
    g_scores = {start: 0}
    return rbfs(graph, start, goal, float('inf'), [start], g_scores)[0]

# Execute RBFS
start = 'Mumbai'
goal = 'Kolhapur'

start_time = time.time()
rbfs_result = recursive_best_first_search(G, start, goal)
rbfs_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using RBFS:", rbfs_result)
print("RBFS Time:", rbfs_time, "seconds")

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

# Add city markers
for city, (lat, lon) in city_coordinates.items():
    folium.Marker(
        location=[lat, lon],
        popup=city,
    ).add_to(m)

# Add paths
if rbfs_result:
    for i in range(len(rbfs_result) - 1):
        folium.PolyLine(
            locations=[city_coordinates[rbfs_result[i]], city_coordinates[rbfs_result[i+1]]],
            color='green',
            weight=2.5,
            opacity=0.7
        ).add_to(m)

# Save the map
m.save('maharashtra_rbfs_map.html')

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')
plt.show()

Algorithm:
1.Define City Data: Create dictionaries for city connections (distances) and city coordinates.
2.Create Graph: Use NetworkX to add cities as nodes and connections as edges with distances as weights.
3.Heuristic Function: Calculate the straight-line distance between cities to estimate the cost to the goal.
4.A Search Algorithm*:
Use a priority queue to explore nodes based on the lowest cost (g + heuristic).
Track the path and g-scores to find the shortest route.
5.Recursive Best-First Search (RBFS):
Explore nodes recursively based on their f-scores (g + heuristic).
Expand the node with the lowest f-score and update recursively until the goal is reached.
6.Execute A and RBFS*: Run both algorithms to find the path and measure execution time.
7.Visualization:
Use Folium to create a map displaying the paths found by A* and RBFS.
Plot the graph using NetworkX, showing cities and connections.

Code: Compare
import folium
import networkx as nx
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("Niraj Singh,T113")

# Define city connections and coordinates
city_connections = {
    'Mumbai': {'Pune': 149, 'Nashik': 167, 'Goa': 467},
    'Pune': {'Mumbai': 149, 'Aurangabad': 235, 'Satara': 115},
    'Nagpur': {'Aurangabad': 280},
    'Nashik': {'Mumbai': 167, 'Aurangabad': 199},
    'Aurangabad': {'Pune': 235, 'Nagpur': 280, 'Nashik': 199},
    'Goa': {'Mumbai': 467, 'Kolhapur': 221},
    'Satara': {'Pune': 115, 'Kolhapur': 130},
    'Kolhapur': {'Satara': 130, 'Goa': 221},
}

city_coordinates = {
    'Mumbai': (19.0760, 72.8777),
    'Pune': (18.5204, 73.8567),
    'Nagpur': (21.1458, 79.0882),
    'Nashik': (20.0116, 73.7908),
    'Aurangabad': (19.8762, 75.3433),
    'Goa': (15.2993, 74.1240),
    'Satara': (17.6868, 73.8567),
    'Kolhapur': (16.7054, 74.2198),
}

# Create the graph
G = nx.Graph()

for city, connections in city_connections.items():
    for neighbor, distance in connections.items():
        G.add_edge(city, neighbor, weight=distance)

# Heuristic function (straight-line distance to the goal)
def heuristic(node, goal):
    x1, y1 = city_coordinates[node]
    x2, y2 = city_coordinates[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# A* Search function
def a_star_search(graph, start, goal):
    queue = []
    heapq.heappush(queue, (0, start, [start]))
    visited = set()
    g_scores = {start: 0}

    while queue:
        _, current, path = heapq.heappop(queue)
        if current == goal:
            return path
        
        visited.add(current)
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_scores[current] + graph[current][neighbor]['weight']
            if neighbor not in visited or tentative_g_score < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(queue, (f_score, neighbor, path + [neighbor]))
    
    return None

# Recursive Best-First Search function
def rbfs(graph, current, goal, f_limit, path, g_scores):
    if current == goal:
        return path

    successors = []
    for neighbor in graph.neighbors(current):
        if neighbor not in path:
            g_score = g_scores[current] + graph[current][neighbor]['weight']
            f_score = g_score + heuristic(neighbor, goal)
            successors.append((neighbor, f_score))
            g_scores[neighbor] = g_score
    
    if not successors:
        return None
    
    successors.sort(key=lambda x: x[1])

    while successors:
        best, f_best = successors[0]
        if f_best > f_limit:
            return None
        
        alternative = successors[1][1] if len(successors) > 1 else float('inf')
        result = rbfs(graph, best, goal, min(f_limit, alternative), path + [best], g_scores)
        if result:
            return result
        
        successors.pop(0)
    
    return None

# Helper function for RBFS
def recursive_best_first_search(graph, start, goal):
    g_scores = {start: 0}
    return rbfs(graph, start, goal, float('inf'), [start], g_scores)

# Execute A* Search
start = 'Mumbai'
goal = 'Kolhapur'

start_time = time.time()
a_star_result = a_star_search(G, start, goal)
a_star_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using A* Search:", a_star_result)
print("A* Search Time:", a_star_time, "seconds")

# Execute RBFS
start_time = time.time()
rbfs_result = recursive_best_first_search(G, start, goal)
rbfs_time = time.time() - start_time

print("Path from Mumbai to Kolhapur using RBFS:", rbfs_result)
print("RBFS Time:", rbfs_time, "seconds")

# Create the map with folium
m = folium.Map(location=[19.0760, 72.8777], zoom_start=6)

# Add city markers
for city, (lat, lon) in city_coordinates.items():
    folium.Marker(
        location=[lat, lon],
        popup=city,
    ).add_to(m)

# Add paths
for path in [a_star_result, rbfs_result]:
    if path:
        for i in range(len(path) - 1):
            folium.PolyLine(
                locations=[city_coordinates[path[i]], city_coordinates[path[i+1]]],
                color='blue' if path == a_star_result else 'green',
                weight=2.5,
                opacity=0.7
            ).add_to(m)

# Save the map
m.save('maharashtra_graph_map.html')

# Plot the graph using networkx
plt.figure(figsize=(10, 8))
pos = {city: (lon, lat) for city, (lat, lon) in city_coordinates.items()}
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_color='black', font_weight='bold')
plt.show()

 
Practical 3

Aim:
Decision Tree Learning 
Implement the Decision Tree Learning algorithm to build a decision tree for a given dataset. 
Evaluate the accuracy and effectiveness of the decision tree on test data. 
Visualize and interpret the generated decision tree.

Algorithm:
1.Import Libraries: Load pandas, matplotlib, and sklearn.
2.Load Dataset: Read data from Salary_Data.csv.
3.Data Preprocessing:
Replace non-numeric values with NaN.
Drop rows with NaN values.
4.Prepare Data:
Select features (Years of Experience) and target variable (Salary).
Convert salary to binary classification based on a median threshold.
5.Split Dataset: Divide data into training (70%) and testing (30%) sets.
6.Initialize Classifier: Create a DecisionTreeClassifier.
7.Train Model: Fit the classifier on the training set.
8.Make Predictions: Predict and evaluate accuracy on the test set.
9.Evaluate Performance: Print accuracy and classification reports.
10.ROC Curve: Calculate and plot the ROC curve.
11.Visualize Tree: Plot the initial decision tree.
12.Pruning:
Determine optimal ccp_alpha for pruning.
Select a target alpha based on impurity plot.
13.Train Pruned Tree: Fit a new classifier with selected ccp_alpha.
14.Evaluate Pruned Model: Predict and print accuracy for the pruned tree.
15.Visualize Pruned Tree: Plot the pruned decision tree.
16.Node Count Check: Print the number of nodes for both trees.
Code:
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

print("Niraj Singh,T113")

# Load the Salary Data dataset
df = pd.read_csv('Salary_Data.csv')  # Replace with the correct filename

# Display column names to verify
print("Column Names:")
print(df.columns)

# Handle non-numeric values and drop rows with NaN values if any
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Prepare features and target variable for classification
X = df[['Years of Experience']]  # Feature
y = df['Salary']

# Convert continuous salary to binary classification
# Define a threshold for binary classification
threshold = df['Salary'].median()  # Use the median salary as the threshold
y_binary = (y > threshold).astype(int)  # 1 if salary is above threshold, 0 otherwise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {test_accuracy:.2f}")

# Evaluate the model's performance on the training set
y_train_pred = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Set Accuracy: {train_accuracy:.2f}")

# Classification Report
print("\nTest Dataset Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTraining Dataset Classification Report:")
print(classification_report(y_train, y_train_pred))

# ROC Curve
y_prob = classifier.predict_proba(X_test)[:, 1]  # Probability estimates for positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Visualizing the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(classifier, feature_names=['Years of Experience'], class_names=['Low', 'High'], filled=True, rounded=True)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()

# Finding Optimal Alpha for Pruning
path = classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Plotting Total Impurity vs Effective Alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities, marker='o', drawstyle="steps-post")
plt.xlabel("Effective Alpha")
plt.ylabel("Total Impurity")
plt.title("Total Impurity vs Effective Alpha for Training Set")
plt.grid(True)
plt.show()

# Manually select a ccp_alpha value to achieve a pruned tree with fewer nodes
# This value may need adjustment based on the output of the above plot
target_alpha = 0.01  # Adjust this value based on your needs

# Initialize the Pruned Decision Tree Classifier
pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=target_alpha)

# Train the pruned model
pruned_tree.fit(X_train, y_train)

# Accuracy After Pruning
pruned_y_pred = pruned_tree.predict(X_test)
pruned_test_accuracy = accuracy_score(y_test, pruned_y_pred)
pruned_train_accuracy = accuracy_score(y_train, pruned_tree.predict(X_train))

print(f"\nPruned Test Set Accuracy: {pruned_test_accuracy:.2f}")
print(f"Pruned Training Set Accuracy: {pruned_train_accuracy:.2f}")

# Visualizing the Pruned Decision Tree
plt.figure(figsize=(12, 6))  # Adjusted figure size for smaller tree
plot_tree(pruned_tree, feature_names=['Years of Experience'], class_names=['Low', 'High'], filled=True, rounded=True)
plt.title("Pruned Decision Tree Visualization", fontsize=18)  # Increased font size for better visibility
plt.show()

# Check the number of nodes in both trees
print(f"Original Decision Tree Node Count: {classifier.tree_.node_count}")
print(f"Pruned Decision Tree Node Count: {pruned_tree.tree_.node_count}")
 
Practical 4

Aim:
Feed Forward Backpropagation Neural Network 
Implement the Feed Forward Backpropagation algorithm to train a neural network. 
Use a given dataset to train the neural network for a specific task. 
Evaluate the performance of the trained network on test data.

Algorithm:
1.Import Libraries: Load necessary libraries (pandas, matplotlib, scikit-learn).
2.Load Data: Load the salary dataset from a CSV file.
3.Handle Missing Data: Replace invalid values and drop rows with missing data.
4.Prepare Features and Target: Use "Years of Experience" as features (X) and "Salary" as target (y).
5.Binary Classification: Convert "Salary" into binary classification using the median as the threshold.
6.Split Data: Split data into training and testing sets (70% training, 30% testing).
7.Initialize MLP Classifier: Set up a neural network with 1 hidden layer (10 neurons) and a ReLU activation function.
8.Train Model: Train the model on the training data.
9.Make Predictions: Predict salary classifications on the test set.
10.Evaluate Model: Calculate accuracy, classification report, and print results.
11.Confusion Matrix: Display confusion matrix to evaluate prediction performance.
12.Dataset Shape: Print the shape of features and labels.
13.Plot Loss Curve: Plot the training loss curve to visualize training progress.

Code:
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Niraj Singh, T113")

# Load the Salary Data dataset
df = pd.read_csv('Salary_Data.csv')  # Replace with the correct filename

# Display column names to verify
print("Column Names:")
print(df.columns)

# Handle non-numeric values and drop rows with NaN values if any
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Prepare features and target variable for classification
X = df[['Years of Experience']]  # Feature
y = df['Salary']

# Convert continuous salary to binary classification
# Define a threshold for binary classification
threshold = df['Salary'].median()  # Use the median salary as the threshold
y_binary = (y > threshold).astype(int)  # 1 if salary is above threshold, 0 otherwise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Initialize the MLP Classifier (Feed Forward Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, activation='relu', solver='adam', random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp.predict(X_test)

# Evaluate the model's performance on the test set
test_accuracy = mlp.score(X_test, y_test)
print(f"\nTest Set Accuracy: {test_accuracy:.2f}")

# Evaluate the model's performance on the training set
train_accuracy = mlp.score(X_train, y_train)
print(f"\nTraining Set Accuracy: {train_accuracy:.2f}")

# Classification Report
print("\nTest Dataset Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTraining Dataset Classification Report:")
print(classification_report(y_train, mlp.predict(X_train)))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Low', 'High'])

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title("Confusion Matrix")
plt.show()

# Shape of the dataset
print("\nShape of the Dataset:")
print(f"Features: {X.shape}")
print(f"Labels: {y_binary.shape}")

# Plot Training Progress (Loss Curve)
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()
 
Practical 5

Aim:
Support Vector Machines (SVM) 
Implement the SVM algorithm for binary classification. 
Train an SVM model using a given dataset and optimize its parameters. 
Evaluate the performance of the SVM model on test data and analyze the results.

Algorithm:
1.Load Data: Import salary data from a CSV file.
2.Preprocess Data: Replace missing values, drop incomplete rows.
3.Prepare Features/Labels: Use "Years of Experience" as input and convert "Salary" into a binary classification (high/low based on median).
4.Split Data: Divide into training (70%) and testing (30%) sets.
5.Normalize Data: Scale the input feature for better performance.
6.Set Hyperparameters: Define a grid of values for C, gamma, and kernel for the SVM model.
7.Grid Search: Perform cross-validation to find the best hyperparameters.
8.Train Model: Train the SVM with the best parameters.
9.Predict: Make predictions on the test set.
10.Evaluate: Measure accuracy, print classification report, and show confusion matrix.
11.Visualize: Display the confusion matrix and a scatter plot of classification results.

Code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

print("Niraj Singh,T113")

# Load and Prepare the Data
df = pd.read_csv('Salary_Data.csv')  # Replace with the correct filename

# Display column names to verify
print("Column Names:")
print(df.columns)

# Handle non-numeric values and drop rows with NaN values if any
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Prepare features and target variable for classification
X = df[['Years of Experience']]  # Feature
y = df['Salary']

# Convert continuous salary to binary classification
threshold = df['Salary'].median()  # Use the median salary as the threshold
y_binary = (y > threshold).astype(int)  # 1 if salary is above threshold, 0 otherwise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Initialize the SVM model
svm = SVC()

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_svm = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Make predictions with the best model
y_pred = best_svm.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Below Threshold', 'Above Threshold'], yticklabels=['Below Threshold', 'Above Threshold'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of the SVM Model")
plt.show()

# Optional: Visualizing the SVM classification results (for 2D)
# For 2D visualization, we'll create synthetic features for demonstration purposes
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, c=y_pred, cmap='coolwarm', marker='o', edgecolor='k')
plt.title('SVM Classification Results')
plt.xlabel('Years of Experience')
plt.ylabel('Binary Salary Classification')
plt.show()

 
Practical 6
Adaboost Ensemble Learning 
Implement the Adaboost algorithm to create an ensemble of weak classifiers.
Train the ensemble model on a given dataset and evaluate its performance.
Compare the results with individual weak classifiers.

Algorithm:
1.Import Libraries: Import pandas, numpy, matplotlib, and necessary sklearn functions.
2.Load and Preprocess Data:
Load dataset from Salary_Data.csv.
Drop rows with NaN values in the Salary column.
Define features and one-hot encode categorical variables.
3.Define Target Variable:
Categorize Salary into bins (Low, Medium, High, Very High) and create Salary_Category.
4.Split Dataset: Split data into training (80%) and testing (20%) sets.
5.Train Weak Classifier:
Initialize and fit a Decision Tree Classifier.
Perform 5-fold cross-validation and evaluate accuracy.
6.Evaluate Weak Classifier:
Calculate and display confusion matrix and classification report.
7.Train AdaBoost Classifier:
Initialize and fit an AdaBoost Classifier.
Perform 5-fold cross-validation and evaluate accuracy.
8.Evaluate AdaBoost Classifier:
Calculate and display confusion matrix and classification report.
9.Visualize Results: Create a bar graph comparing the accuracies of both classifiers and display it.

Model Selection Method:
1.Cross-Validation Accuracy: Evaluates model performance across multiple data splits to get a reliable average accuracy.
2.Confusion Matrix: Provides a breakdown of true/false positives and negatives to assess classification accuracy.
3.Classification Report: Summarizes precision, recall, and F1-score to measure the model's effectiveness.
4.Accuracy Comparison (Bar Graph): Visualizes the overall accuracy of each model to compare their performance.

Code:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Niraj Singh,T113")

# Load your dataset
salary_df = pd.read_csv('Salary_Data.csv')  # Update with your dataset path

# Handle NaN values in the Salary column (e.g., drop them)
salary_df = salary_df.dropna(subset=['Salary'])

# Define your features and target variable
feature_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
X = salary_df[feature_columns]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Define target variable and categorize salaries
bins = [0, 60000, 100000, 150000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High']
salary_df['Salary_Category'] = pd.cut(salary_df['Salary'], bins=bins, labels=labels)

# Use the salary category as the target variable for classification
y_classification = salary_df['Salary_Category']

# Check for NaN values in target variable
print("\nNaN values in target variable y_classification:", y_classification.isna().sum())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Proceed with the classification model training
classifier = DecisionTreeClassifier()

# Perform 5-fold cross-validation for the weak classifier
cv_scores_classifier = cross_val_score(classifier, X_train, y_train, cv=5)
print("\nWeak Classifier Cross-Validation Accuracy:", cv_scores_classifier.mean())
print("Weak Classifier Cross-Validation Scores:", cv_scores_classifier)

# Fit the classification model on the training dataset
classifier.fit(X_train, y_train)

# Make predictions for classification
y_pred_classification = classifier.predict(X_test)

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classification)
print("\nConfusion Matrix for Weak Classifier:\n", conf_matrix)

# Display the confusion matrix using a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Weak Classifier)')
plt.show()

# Classification report
print("\nClassification Report for Weak Classifier:\n", classification_report(y_test, y_pred_classification))

# Proceed with AdaBoost classifier training
adaboost_classifier = AdaBoostClassifier(n_estimators=50, algorithm='SAMME')  # Specified algorithm as 'SAMME'

# Perform 5-fold cross-validation for AdaBoost
cv_scores_adaboost = cross_val_score(adaboost_classifier, X_train, y_train, cv=5)
print("AdaBoost Classifier Cross-Validation Accuracy:", cv_scores_adaboost.mean())
print("AdaBoost Classifier Cross-Validation Scores:", cv_scores_adaboost)

# Fit the AdaBoost classifier on the training dataset
adaboost_classifier.fit(X_train, y_train)

# Make predictions for AdaBoost classification
y_pred_adaboost = adaboost_classifier.predict(X_test)

# Calculate and display confusion matrix for AdaBoost
conf_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)
print("\nConfusion Matrix for AdaBoost:\n", conf_matrix_adaboost)

# Display the confusion matrix using a heatmap
disp_adaboost = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_adaboost, display_labels=adaboost_classifier.classes_)
disp_adaboost.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (AdaBoost Classifier)')
plt.show()

# Classification report for AdaBoost
print("\nClassification Report for AdaBoost Classifier:\n", classification_report(y_test, y_pred_adaboost))

# Bar graph comparison of classification accuracy
accuracy_scores = [cv_scores_classifier.mean(), cv_scores_adaboost.mean()]
labels = ['Weak Classifier', 'AdaBoost']
plt.bar(labels, accuracy_scores, color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Comparison of Classifier Accuracy')
plt.ylim(0, 1)
plt.show()

 
Practical 7
Naive Bayes' Classifier
Implement the Naive Bayes' algorithm for classification.
Train a Naive Bayes' model using a given dataset and calculate class probabilities.
Evaluate the accuracy of the model on test data and analyse the results.

Algorithm:
1.Load Data: Import the dataset and handle missing values.
2.Define Features: Set X (input) and y (target).
3.Binarize Target: Convert y to binary based on the median salary.
4.Split Data: Divide data into training and testing sets.
5.Train Models: Use Gaussian and Bernoulli Naive Bayes classifiers.
6.Predict: Generate predictions on the test set.
7.Evaluate: Calculate accuracy, generate classification reports, and confusion matrices.
8.Visualize: Plot confusion matrices and accuracy comparisons.

Equation Explanation:
Write Key Equations until being 1.
 

Code:
# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Niraj Singh,T113")

# Load the dataset
df = pd.read_csv('Salary_Data.csv')  

# Handle missing values
df.dropna(inplace=True)

# Define features and target variable
X = df[['Years of Experience']]  
y = df['Salary']  # Target variable

# Binarize target variable (above or below median salary)
threshold = df['Salary'].median()
y = (y > threshold).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Gaussian Naive Bayes classifier
gnb_model = GaussianNB()

# Train the model
gnb_model.fit(X_train, y_train)

# Predict using Gaussian Naive Bayes
y_pred_gnb = gnb_model.predict(X_test)

# Evaluate accuracy for GaussianNB
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print(f'\nGaussianNB Accuracy: {accuracy_gnb:.2f}')

# Classification report for GaussianNB
print("\nGaussianNB Classification Report:")
print(classification_report(y_test, y_pred_gnb))

# Confusion matrix for GaussianNB
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
print("\nGaussianNB Confusion Matrix:")
print(cm_gnb)

# Plot confusion matrix for GaussianNB
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', xticklabels=['Below Median', 'Above Median'], yticklabels=['Below Median', 'Above Median'])
plt.title('Confusion Matrix for Gaussian Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Initialize Bernoulli Naive Bayes classifier
bnb_model = BernoulliNB()

# Train the model
bnb_model.fit(X_train, y_train)

# Predict using Bernoulli Naive Bayes
y_pred_bnb = bnb_model.predict(X_test)

# Evaluate accuracy for BernoulliNB
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)
print(f'\nBernoulliNB Accuracy: {accuracy_bnb:.2f}')

# Classification report for BernoulliNB
print("\nBernoulliNB Classification Report:")
print(classification_report(y_test, y_pred_bnb))

# Confusion matrix for BernoulliNB
cm_bnb = confusion_matrix(y_test, y_pred_bnb)
print("\nBernoulliNB Confusion Matrix:")
print(cm_bnb)

# Plot confusion matrix for BernoulliNB
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bnb, annot=True, fmt='d', cmap='Blues', xticklabels=['Below Median', 'Above Median'], yticklabels=['Below Median', 'Above Median'])
plt.title('Confusion Matrix for Bernoulli Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Accuracy comparison bar graph
models = ['GaussianNB', 'BernoulliNB']
accuracies = [accuracy_gnb, accuracy_bnb]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=accuracies, hue=models, dodge=False, palette='viridis', legend=False)
plt.title('Accuracy Comparison of Naive Bayes Classifiers')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits to [0, 1]
plt.show()

 
Practical 8
Aim:
K-Nearest Neighbors (K-NN)
Implement the K-NN algorithm for classification or regression.
Apply the K-NN algorithm to a given dataset and predict the class or value for test data.
Evaluate the accuracy or error of the predictions and analyze the results.

Algorithm:
1.Import Libraries: Load necessary libraries like pandas, numpy, sklearn, and plotting tools.
2.Load Dataset: Read the Salary_Data.csv file and display column names.
3.Handle Missing Values: Impute missing values in Age, Years of Experience, and Salary with the mean.
4.Encode Categorical Features: Convert Gender to numerical format using label encoding.
5.Categorize Salary: Divide Salary into three bins: Low, Medium, High.
6.Define Features and Target: Set Age, Gender, and Years of Experience as features (X), and Salary Category as the target (y).
7.Split Dataset: Split data into training (80%) and testing (20%) sets.
8.Standardize Features: Scale the features using StandardScaler.
9.Train K-NN Classifier: Loop over different k-values (e.g., 3, 5, 7, 9), train K-NN, predict, calculate accuracy, and print results.
10.Plot Decision Boundaries: Visualize decision boundaries for Age and Years of Experience.
11.Visualize Accuracy: Plot accuracy for different k-values.

Code:
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Salary_Data.csv')

# Display the column names of the dataset
print("Column Names:", df.columns)

# Handle missing values by imputing with the mean (for numeric columns)
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Years of Experience'] = imputer.fit_transform(df[['Years of Experience']])
df['Salary'] = imputer.fit_transform(df[['Salary']])

# Encode categorical features if necessary (Gender)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

# Categorize the Salary into bins (low, medium, high)
df['Salary Category'] = pd.cut(df['Salary'], bins=3, labels=['Low', 'Medium', 'High'])

# Select features (Age, Gender, Years of Experience) and target (Salary Category)
X = df[['Age', 'Gender', 'Years of Experience']]
y = df['Salary Category']

# Convert target variable to numerical format
y = y.cat.codes  # Convert categorical labels to numerical codes

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (Important for K-NN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize arrays to store accuracy for different k-values
k_values = [3, 5, 7, 9]
accuracies = []
confusion_matrices = []

# Loop over k-values
for k in k_values:
    # Initialize and fit the KNN model for classification
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Predict the target variable (salary category) for test data
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])  # Use numerical codes for labels
    confusion_matrices.append(cm)
    
    # Print confusion matrix, classification report, and accuracy
    print(f'\nConfusion Matrix for K={k}:')
    print(cm)
    print(f'\nClassification Report for K={k}:')
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    print(f'Accuracy for K={k}: {accuracy * 100:.2f}%')
    
    # Plot the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(f'Confusion Matrix for K={k}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plotting decision boundaries for Age and Years of Experience
    plt.figure(figsize=(8, 6))
    
    # Create a meshgrid for Age and Years of Experience
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1  # Age
    y_min, y_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1  # Years of Experience
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Create a grid of points with Gender set to 0 (for simplicity)
    Z = knn.predict(np.column_stack((xx.ravel(), np.zeros_like(xx.ravel()), yy.ravel())))
    
    # Reshape Z back to the shape of the meshgrid
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # No need to cast Z to float
    plt.scatter(X_train[:, 0], X_train[:, 2], c=y_train, edgecolors='k', marker='o', label='Training data')
    plt.scatter(X_test[:, 0], X_test[:, 2], c=y_test, edgecolors='k', marker='s', label='Test data')
    plt.title(f'Decision Boundary for K={k}')
    plt.xlabel('Age')
    plt.ylabel('Years of Experience')
    plt.legend()
    plt.show()

# Plotting the accuracy vs. k-value graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('K-NN Classification: Accuracy for Different K Values')
plt.show()

 
Practical 9
Aim:
Association Rule Mining
Implement the Association Rule Mining algorithm (e.g., Apriori) to find frequent itemsets.
Generate association rules from the frequent itemsets and calculate their support and confidence.
Interpret and analyze the discovered association rules.

Algorithm:
1. Import Libraries: Load necessary libraries (pandas, mlxtend, matplotlib, seaborn).
2. Load Dataset: Read the CSV file into a DataFrame.
3. Data Preprocessing:
Convert categorical variables (Gender, Education Level, Job Title) to numeric/binary.
Bin Age and Years of Experience into categorical ranges.
4. One-Hot Encoding: Apply one-hot encoding to convert categorical data into binary features.
5. Run Apriori Algorithm: Use the apriori() function to find frequent itemsets with a minimum support threshold.
6. Generate Association Rules: Use association_rules() to extract rules based on confidence.
7. Display Results:
Print frequent itemsets and association rules (with support, confidence, lift).
8. Plot Support vs Confidence: Create a scatterplot to visualize the relationship between support and confidence, with lift as size.
9. Top Rules by Lift & Support: Sort rules by lift and support, and plot bar charts for the top 5 rules.

Code:
# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

print("Niraj Singh, T113")

# Load the dataset
df = pd.read_csv('Salary_Data.csv')

# Data Preprocessing
# Convert categorical variables to binary
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Education Level'] = pd.Categorical(df['Education Level']).codes  # Convert Education Level to numeric codes
df['Job Title'] = pd.Categorical(df['Job Title']).codes  # Convert Job Title to numeric codes

# Create bins for Age and Years of Experience to make the data more categorical for association mining
df['Age_Binned'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])
df['Experience_Binned'] = pd.cut(df['Years of Experience'], bins=[0, 5, 10, 15, 20, 25], labels=['0-5', '5-10', '10-15', '15-20', '20-25'])

# Convert to one-hot encoding for Apriori (this will convert each bin/category into binary features)
df_encoded = pd.get_dummies(df[['Gender', 'Education Level', 'Job Title', 'Age_Binned', 'Experience_Binned']], drop_first=True)

# Ensure all values are binary (0/1) for the apriori function
df_encoded = df_encoded.astype(bool)

# Run Apriori to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Display frequent itemsets
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display association rules
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Evaluation: Plot support vs confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(x="support", y="confidence", size="lift", data=rules)
plt.title("Support vs Confidence with Lift as Size")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# Top 5 rules by lift
top_lift_rules = rules.sort_values(by="lift", ascending=False).head(5)
print("\nTop 5 rules by Lift:")
print(top_lift_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Plot top 5 rules by lift
plt.figure(figsize=(10, 6))
sns.barplot(x=top_lift_rules['lift'], y=top_lift_rules['antecedents'].apply(lambda x: ', '.join(list(x))), hue=top_lift_rules['antecedents'], palette='viridis', legend=False)
plt.title("Top 5 Association Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Antecedents")
plt.show()

# Top 5 rules by support
top_support_rules = rules.sort_values(by="support", ascending=False).head(5)
print("\nTop 5 rules by Support:")
print(top_support_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Plot top 5 rules by support
plt.figure(figsize=(10, 6))
sns.barplot(x=top_support_rules['support'], y=top_support_rules['antecedents'].apply(lambda x: ', '.join(list(x))), hue=top_support_rules['antecedents'], palette='Blues_d', legend=False)
plt.title("Top 5 Association Rules by Support")
plt.xlabel("Support")
plt.ylabel("Antecedents")
plt.show()
