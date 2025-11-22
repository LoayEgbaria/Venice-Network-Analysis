#import osmnx as ox
# Define the place name or address
#place_name = "Venice, Italy"
# Download the street network data for Venice
#graph = ox.graph_from_place(place_name, network_type="all")
# Save the graph as a shapefile
#ox.save_graph_shapefile(graph, filename="venice_network")
#print("Data downloaded successfully!")


import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import infomap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Define the place name or address
#place_name = "Venice, Italy"
place_name = "Venezia-Murano-Burano, Veneto, Italy"

# Download the street network data for Venice
graph = ox.graph_from_place(place_name, network_type="all")
#graph = ox.graph_from_place(place_name, network_type="walk", simplify=True)

# Make sure the output directory exists
output_dir = "VeniceNetworkFiles"
os.makedirs(output_dir, exist_ok=True)

# Save the graph as a GraphML file
ox.save_graphml(graph, filepath="VeniceNetworkFiles/venice.graphml",gephi=True)
print("Data downloaded successfully!")

# Make sure the output directory exists
output_dir = "OutputImages"
os.makedirs(output_dir, exist_ok=True)

# Plot and Save the graph image
ox.plot_graph(graph,filepath="OutputImages/venice_graph.png",save=True,dpi=300)

# Convert to undirected for community detection
G = graph.to_undirected()


# Get largest connected component
if not nx.is_connected(G):
    print("Graph is not connected. Using largest component...")
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"Largest component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ========== Infomap Algorithm ==========
print("\n" + "="*60)
print("Running Infomap Algorithm...")
print("="*60)

try:
    
    # Create Infomap instance
    #im = infomap.Infomap("--two-level --markov-time 150.0")
    #im = infomap.Infomap("--two-level --markov-time 195.0 --seed 42")
    # This automatically runs 10 times and picks best result
    #im = infomap.Infomap("--markov-time 195.0 --num-trials 10")
    
    #im = infomap.Infomap("--two-level --markov-time 50.0 --num-trials 10")
    
    im = infomap.Infomap("--two-level --markov-time 47.0 --seed 42")
    
    #im = infomap.Infomap("--two-level --markov-time 45.0")
    
 
    
    
    
    # Create mapping from original node IDs to sequential integers
    node_list = list(G.nodes())
    node_to_int = {node: i for i, node in enumerate(node_list)}
    int_to_node = {i: node for node, i in node_to_int.items()}
    
    # Add nodes and edges with integer IDs
    for edge in G.edges():
        source_int = node_to_int[edge[0]]
        target_int = node_to_int[edge[1]]
        im.add_link(source_int, target_int)
    
    # Run algorithm
    im.run()
    
    print(f"Infomap found {im.num_top_modules} communities")
    print(f"Codelength: {im.codelength:.4f}")
    
    # Extract communities and map back to original node IDs
    infomap_communities_int = {}
    for node in im.tree:
        if node.is_leaf:
            infomap_communities_int[node.node_id] = node.module_id
    
    # Convert integer IDs back to original node IDs
    infomap_communities = {
        int_to_node[node_int]: module 
        for node_int, module in infomap_communities_int.items()
    }
    
    # Convert to list of sets (standard format)
    unique_modules = set(infomap_communities.values())
    infomap_comm_list = [
        set([node for node, mod in infomap_communities.items() if mod == module])
        for module in unique_modules
    ]
    
    # Calculate modularity
    infomap_modularity = nx.algorithms.community.modularity(G, infomap_comm_list)
    print(f"Modularity: {infomap_modularity:.4f}")
    
    # Community sizes
    sizes = [len(comm) for comm in infomap_comm_list]
    print(f"Community sizes: {sorted(sizes, reverse=True)[:10]}")
    
    infomap_available = True
    
except ImportError:
    print("Warning: 'infomap' package not installed.")
    print("Install with: pip install infomap")
    infomap_available = False
    exit()

# ========== ENHANCED INFOMAP VISUALIZATIONS ==========
print("\n" + "="*60)
print("Creating Enhanced Infomap Visualizations...")
print("="*60)

# Get node positions from graph
pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# Create node to community mapping
node_to_comm = {}
for i, comm in enumerate(infomap_comm_list):
    for node in comm:
        node_to_comm[node] = i

# ========== VISUALIZATION 1: Basic Infomap Communities ==========
print("Creating basic community visualization...")

plt.figure(figsize=(14, 10))

# Prepare colors for communities
n_communities = len(infomap_comm_list)
colors = plt.cm.tab20(np.linspace(0, 1, n_communities))

# Draw each community with different color
for i, comm in enumerate(infomap_comm_list):
    comm_nodes = list(comm)
    nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes, 
                          node_color=[colors[i]], 
                          node_size=30, alpha=0.8, 
                          label=f'Community {i+1} ({len(comm)} nodes)')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, edge_color='gray')

plt.title(f'Infomap Communities - Venice Network\n'
         f'{n_communities} communities, Modularity: {infomap_modularity:.3f}',
         fontsize=14, fontweight='bold')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.axis('equal')
plt.tight_layout()

save_path = os.path.join("OutputImages", "venice_infomap_communities.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

#plt.savefig(save_path, dpi=300, bbox_inches="tight")
#plt.savefig("venice_infomap_communities.png", dpi=300)  # save PNG with high resolution

plt.show()

# ========== VISUALIZATION 2: Detailed Multi-Panel View ==========
print("Creating detailed multi-panel visualization...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Infomap Community Detection - Detailed Analysis', fontsize=16, fontweight='bold')

# Panel 1: Communities with nodes colored
ax1 = axes[0, 0]
node_colors = [node_to_comm.get(node, -1) for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=25,
                      cmap='tab20', alpha=0.8, ax=ax1)
nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.4, ax=ax1)
ax1.set_title(f'{n_communities} Communities (Modularity: {infomap_modularity:.3f})')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.axis('equal')

# Panel 2: Community sizes distribution
ax2 = axes[0, 1]
comm_sizes = sorted([len(comm) for comm in infomap_comm_list], reverse=True)
#ax2.bar(range(len(comm_sizes)), comm_sizes, color='steelblue', alpha=0.7)

ax2.bar([i + 1 for i in range(len(comm_sizes))], comm_sizes, color='steelblue', alpha=0.7)

ax2.set_xlabel('Community Index (sorted by size)')
ax2.set_ylabel('Number of Nodes')
ax2.set_title('Community Size Distribution')
ax2.grid(True, alpha=0.3)

# Panel 3: Top 5 largest communities highlighted
ax3 = axes[1, 0]
# Sort communities by size
sorted_comms = sorted(enumerate(infomap_comm_list), key=lambda x: len(x[1]), reverse=True)

# Draw all nodes in gray first
nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=15, 
                      alpha=0.3, ax=ax3)
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.3, edge_color='gray', ax=ax3)

# Highlight top 5 communities
top5_colors = plt.cm.Set1(np.linspace(0, 1, 5))
legend_patches = []
for rank, (comm_idx, comm) in enumerate(sorted_comms[:5]):
    comm_nodes = list(comm)
    nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes,
                          node_color=[top5_colors[rank]], 
                          node_size=30, alpha=0.9, ax=ax3)
    patch = mpatches.Patch(color=top5_colors[rank], 
                          label=f'#{rank+1}: {len(comm)} nodes')
    legend_patches.append(patch)

ax3.set_title('Top 5 Largest Communities')
ax3.legend(handles=legend_patches, loc='upper right')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.axis('equal')

# Panel 4: Community geographic spread
ax4 = axes[1, 1]
spreads = []
for i, comm in enumerate(infomap_comm_list):
    nodes_list = list(comm)
    coords = np.array([pos[node] for node in nodes_list])
    lon_range = coords[:, 0].max() - coords[:, 0].min()
    lat_range = coords[:, 1].max() - coords[:, 1].min()
    geographic_spread = np.sqrt(lon_range**2 + lat_range**2)
    spreads.append(geographic_spread)

ax4.scatter(comm_sizes, spreads, c=range(len(comm_sizes)), 
           cmap='viridis', s=100, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Community Size (nodes)')
ax4.set_ylabel('Geographic Spread')
ax4.set_title('Community Size vs Geographic Spread')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join("OutputImages", "Infomap Community Detection-Detailed Analysis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# ========== VISUALIZATION 3: Individual Large Communities ==========
print("Creating individual community visualizations...")

# Get top 6 largest communities
sorted_comms = sorted(enumerate(infomap_comm_list), key=lambda x: len(x[1]), reverse=True)
n_to_plot = min(6, len(sorted_comms))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Individual Community Visualizations (Largest 6)', fontsize=16, fontweight='bold')
axes = axes.flatten()

for i in range(n_to_plot):
    comm_idx, comm = sorted_comms[i]
    ax = axes[i]
    
    # Draw all nodes in light gray
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', 
                          node_size=10, alpha=0.2, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=0.3, 
                          edge_color='lightgray', ax=ax)
    
    # Highlight this community
    comm_nodes = list(comm)
    nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes,
                          node_color='red', node_size=40, 
                          alpha=0.8, ax=ax)
    
    # Draw edges within community
    subgraph = G.subgraph(comm_nodes)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=1.0,
                          edge_color='red', ax=ax)
    
    # Get geographic bounds
    coords = np.array([pos[node] for node in comm_nodes])
    
    ax.set_title(f'Community {comm_idx+1}\n{len(comm)} nodes, '
                f'{subgraph.number_of_edges()} edges',
                fontsize=10)
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    ax.axis('equal')

# Hide unused subplots
for i in range(n_to_plot, 6):
    axes[i].axis('off')

plt.tight_layout()
save_path = os.path.join("OutputImages", "Individual Community Visualizations.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# ========== VISUALIZATION 4: Heatmap-style Density View ==========
print("Creating density heatmap visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Community Density Analysis', fontsize=16, fontweight='bold')

# Left: Node density by community
coords_list = []
colors_list = []
for node in G.nodes():
    x, y = pos[node]
    comm = node_to_comm.get(node, -1)
    coords_list.append([x, y])
    colors_list.append(comm+1)

coords_array = np.array(coords_list)

scatter1 = ax1.scatter(coords_array[:, 0], coords_array[:, 1], 
                      c=colors_list, cmap='tab20', s=50, alpha=0.7)
ax1.set_title('Nodes Colored by Community')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.axis('equal')
plt.colorbar(scatter1, ax=ax1, label='Community ID')

# Right: Geographic density
from scipy.spatial import cKDTree
tree = cKDTree(coords_array)
densities = []
radius = 0.001  # Adjust based on your coordinate scale

for coord in coords_array:
    nearby = tree.query_ball_point(coord, r=radius)
    densities.append(len(nearby))

scatter2 = ax2.scatter(coords_array[:, 0], coords_array[:, 1], 
                      c=densities, cmap='YlOrRd', s=50, alpha=0.7)
ax2.set_title('Node Density (Geographic)')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.axis('equal')
plt.colorbar(scatter2, ax=ax2, label='Local Density')

plt.tight_layout()
save_path = os.path.join("OutputImages", "Community Density Analysis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# ========== Community Statistics ==========
print("\n" + "="*60)
print("COMMUNITY STATISTICS")
print("="*60)

for i, (comm_idx, comm) in enumerate(sorted_comms[:6]):
    nodes_list = list(comm)
    coords = np.array([pos[node] for node in nodes_list])
    
    print(f"\nCommunity {comm_idx+1} (Rank #{i+1}):")
    print(f"  Size: {len(comm)} nodes")
    print(f"  Geographic center: ({coords[:, 0].mean():.6f}, {coords[:, 1].mean():.6f})")
    print(f"  Geographic bounds:")
    print(f"    Longitude: [{coords[:, 0].min():.6f}, {coords[:, 0].max():.6f}]")
    print(f"    Latitude:  [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]")
    
    # Network statistics for this community
    subgraph = G.subgraph(nodes_list)
    print(f"  Network statistics:")
    print(f"    Edges: {subgraph.number_of_edges()}")
    print(f"    Density: {nx.density(subgraph):.4f}")
    if len(nodes_list) > 1:
        try:
            print(f"    Avg clustering: {nx.average_clustering(subgraph):.4f}")
        except:
            pass

# ========== Save visualization data ==========
print("\n" + "="*60)
print("Saving Results...")
print("="*60)

# Save community assignments
results_df = pd.DataFrame({
    'node_id': list(G.nodes()),
    'longitude': [pos[node][0] for node in G.nodes()],
    'latitude': [pos[node][1] for node in G.nodes()],
    'infomap_community': [node_to_comm.get(node, -1) for node in G.nodes()]
})

results_df.to_csv("VeniceNetworkFiles/venice_infomap_communities.csv", index=False)
print("Community assignments saved to: VeniceNetworkFiles: venice_infomap_communities.csv")

# Save community statistics
comm_stats = []
for i, comm in enumerate(infomap_comm_list):
    nodes_list = list(comm)
    coords = np.array([pos[node] for node in nodes_list])
    subgraph = G.subgraph(nodes_list)
    
    comm_stats.append({
        'community_id': i,
        'size': len(comm),
        'num_edges': subgraph.number_of_edges(),
        'density': nx.density(subgraph),
        'center_lon': coords[:, 0].mean(),
        'center_lat': coords[:, 1].mean(),
        'lon_min': coords[:, 0].min(),
        'lon_max': coords[:, 0].max(),
        'lat_min': coords[:, 1].min(),
        'lat_max': coords[:, 1].max()
    })

stats_df = pd.DataFrame(comm_stats)
#this line to save the community statistics file 
stats_df.to_csv("VeniceNetworkFiles/venice_infomap_community_stats.csv", index=False)
print("Community statistics saved to: VeniceNetworkFiles: venice_infomap_community_stats.csv")

print("\nAll visualizations and analysis complete!")





#============================================================================================#

'''
# ========== METHOD 2: Girvan-Newman Algorithm ==========
print("\n" + "="*60)
print("Running Girvan-Newman Algorithm...")
print("="*60)

# Note: Girvan-Newman is computationally expensive for large graphs
# We'll limit iterations for practical purposes

from networkx.algorithms.community import girvan_newman

# Run Girvan-Newman (generates hierarchical communities)
print("Computing communities (this may take a while for large graphs)...")

# Get iterator
gn_iterator = girvan_newman(G)

# Extract communities at different levels
gn_communities_levels = []
max_levels = 10  # Limit iterations for large graphs

try:
    for i, communities in enumerate(gn_iterator):
        if i >= max_levels:
            break
        
        comm_list = [set(c) for c in communities]
        modularity = nx.algorithms.community.modularity(G, comm_list)
        
        gn_communities_levels.append({
            'level': i,
            'num_communities': len(comm_list),
            'communities': comm_list,
            'modularity': modularity
        })
        
        print(f"Level {i}: {len(comm_list)} communities, Modularity: {modularity:.4f}")
        
except Exception as e:
    print(f"Stopped at level {i}: {e}")

# Find best level by modularity
best_level = max(gn_communities_levels, key=lambda x: x['modularity'])
gn_comm_list = best_level['communities']
gn_modularity = best_level['modularity']

print(f"\nBest level: {best_level['level']} with {best_level['num_communities']} communities")
print(f"Best modularity: {gn_modularity:.4f}")

# Community sizes
sizes = [len(comm) for comm in gn_comm_list]
print(f"Community sizes: {sorted(sizes, reverse=True)[:10]}")  # Top 10

# ========== Comparison ==========
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

if infomap_available:
    print(f"Infomap:")
    print(f"  Communities: {len(infomap_comm_list)}")
    print(f"  Modularity: {infomap_modularity:.4f}")
    print()

print(f"Girvan-Newman:")
print(f"  Communities: {len(gn_comm_list)}")
print(f"  Modularity: {gn_modularity:.4f}")

# ========== Visualization ==========
print("\n" + "="*60)
print("Creating Visualizations...")
print("="*60)

# Get node positions from graph
pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

# Create color map
def assign_colors(communities, nodes):
    """Assign color index to each node based on community"""
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
    return [node_to_comm.get(node, -1) for node in nodes]

# Setup plot
if infomap_available:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
else:
    fig, axes = plt.subplots(1, 1, figsize=(12, 10))
    axes = [axes]

fig.suptitle('Venice Network Community Detection', fontsize=16, fontweight='bold')

# Plot 2: Girvan-Newman
ax2 = axes[-1]
node_colors = assign_colors(gn_comm_list, G.nodes())

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=20,
                      cmap='tab20', alpha=0.8, ax=ax2)
nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax2)

ax2.set_title(f'Girvan-Newman Algorithm (Level {best_level["level"]})\n'
              f'{len(gn_comm_list)} communities, Modularity: {gn_modularity:.3f}',
              fontsize=12)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.axis('equal')

plt.tight_layout()
plt.show()

# ========== Detailed Community Analysis ==========
print("\n" + "="*60)
print("Detailed Community Analysis")
print("="*60)

def analyze_communities(communities, name):
    """Analyze and print community statistics"""
    print(f"\n{name}:")
    print("-" * 40)
    
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:5]):
        nodes_list = list(comm)
        coords = np.array([pos[node] for node in nodes_list])
        
        print(f"\nCommunity {i} (Size: {len(comm)} nodes):")
        print(f"  Geographic bounds:")
        print(f"    Longitude: {coords[:, 0].min():.6f} to {coords[:, 0].max():.6f}")
        print(f"    Latitude:  {coords[:, 1].min():.6f} to {coords[:, 1].max():.6f}")
        print(f"  Geographic spread:")
        print(f"    Lon range: {coords[:, 0].max() - coords[:, 0].min():.6f}")
        print(f"    Lat range: {coords[:, 1].max() - coords[:, 1].min():.6f}")

if infomap_available:
    analyze_communities(infomap_comm_list, "INFOMAP - Top 5 Communities")

analyze_communities(gn_comm_list, "GIRVAN-NEWMAN - Top 5 Communities")

# ========== Save Results ==========
print("\n" + "="*60)
print("Saving Results...")
print("="*60)

# Create node mapping
node_data = []
for node in G.nodes():
    x, y = pos[node]
    
    data = {
        'node_id': node,
        'longitude': x,
        'latitude': y
    }
    
    if infomap_available:
        # Find Infomap community
        for i, comm in enumerate(infomap_comm_list):
            if node in comm:
                data['infomap_community'] = i
                break
    
    # Find Girvan-Newman community
    for i, comm in enumerate(gn_comm_list):
        if node in comm:
            data['girvan_newman_community'] = i
            break
    
    node_data.append(data)

results_df = pd.DataFrame(node_data)
results_df.to_csv("venice_community_detection_results.csv", index=False)

print("Results saved to: venice_community_detection_results.csv")
print(f"Total nodes analyzed: {len(results_df)}")

print("\nAnalysis complete!")
'''

#============================================================================================#




















# REST OF THE CODE FOR: KNN AND SVM ALGORITHMS FOR DENSITY BASED EDGE FILTERING

'''

# Plot the graph
ox.plot_graph(graph, node_size=5, node_color="white", bgcolor="black", edge_color="gray")


# Convert graph to node dataframe
nodes, edges = ox.graph_to_gdfs(graph)


# Save as CSV/TXT files
nodes.to_csv("venice_nodes.txt", index=True)
edges.to_csv("venice_edges.txt", index=True)

print("Data saved as TXT files!")


###############################
import networkx as nx
import matplotlib.pyplot as plt
G_core = graph.copy()
iteration = 0
while True:
    # Find dead-end nodes (degree 1)
    dead_end_nodes = [node for node, degree in dict(G_core.degree()).items() if degree == 1]
    
    if not dead_end_nodes:
        break  # stop when no more dead-ends

    # Remove edges connected to dead-end nodes
    G_core.remove_nodes_from(dead_end_nodes)
    iteration += 1

    print(f"Iteration {iteration}: removed {len(dead_end_nodes)} dead-end nodes")
# -----------------------------
# Step 3: Plot the resulting 2-core graph
# -----------------------------
with open("venice_edges_simple.txt", "w") as f:
    for u, v, data in G_core.edges(data=True):
        f.write(f"{u}\t{v}\n")  # tab-separated

fig, ax = ox.plot_graph(
    G_core,
    node_size=5,
    node_color="red",
    edge_color="gray",
    bgcolor="black"
)
ax.set_title("Venice Network 2-Core (Dead-ends removed)", color="white")
plt.show()
# -----------------------------
# Step 4: Save nodes and edges
# -----------------------------
nodes, edges = ox.graph_to_gdfs(G_core)
nodes.to_csv("venice_nodes_2core.txt")
edges.to_csv("venice_edges_2core.txt")

print(f"Final nodes: {len(nodes)}, Final edges: {len(edges)}")
# Create a copy of the graph
#G_no_oneway = graph.copy()
#edges_to_remove = [(u, v, k) for u, v, k, data in G_no_oneway.edges(keys=True, data=True) if data.get('oneway') == True]
#G_no_oneway.remove_edges_from(edges_to_remove)
#isolated_nodes = list(nx.isolates(G_no_oneway))
#G_no_oneway.remove_nodes_from(isolated_nodes)
#nodes, edges = ox.graph_to_gdfs(G_no_oneway)
#nodes.to_csv("venice_nodes_no_oneway.txt")
#edges.to_csv("venice_edges_no_oneway.txt")
#print("Origin Edges:", len(graph.edges))
#print("Edges after removing oneway:", len(G_no_oneway.edges))
#fig, ax = ox.plot_graph(G_no_oneway,node_size=5,node_color="red",edge_color="gray",bgcolor="black")
#ax.set_title("Venice Network (No Oneway Edges)", color="white")
#plt.show()

# -----------------------------
# Step 2: Project graph to UTM and prepare coordinates
# -----------------------------
G_proj = ox.project_graph(G_core)
nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj)

coords_proj = np.array([(geom.y, geom.x) for geom in nodes_proj.geometry])
tree_proj = cKDTree(coords_proj)
node_ids = list(nodes_proj.index)

radius = 10  # meters; adjust for desired density scale
node_density = {}

for i, node in enumerate(node_ids):
    idxs = tree_proj.query_ball_point(coords_proj[i], r=radius)
    node_density[node] = len(idxs)

threshold = 10  # minimum number of nearby nodes to keep edge
edges_to_remove = []

for u, v, k in G_core.edges(keys=True):
    avg_density = (node_density[u] + node_density[v]) / 2
    if avg_density < threshold:
        edges_to_remove.append((u, v, k))

G_dense_geo = G_core.copy()
G_dense_geo.remove_edges_from(edges_to_remove)

fig, ax = ox.plot_graph(
    G_dense_geo,
    node_size=10,
    node_color="red",
    edge_color="gray",
    bgcolor="black"
)
ax.set_title("Venice Network (Edges in High-Density Areas Only)", color="white")
plt.show()













# Filter nodes with degree >= 2 (exclude dead ends)
filtered_nodes = nodes[nodes['street_count'] >= 2]
filtered_nodes.to_csv("venice_nodes_filtered.txt", index=True)

import matplotlib.pyplot as plt
# Plot just the filtered nodes
fig, ax = plt.subplots(figsize=(8, 8))
filtered_nodes.plot(ax=ax, color="red", markersize=5)
ax.set_facecolor("black")
ax.set_title("Filtered Nodes (degree ≥ 2)", color="white")
plt.show()



# Save filtered nodes in your specified format
with open("venice_nodes_filtered_indexed.txt", "w") as f:
    for i, (node_id, node_data) in enumerate(filtered_nodes.iterrows(), 1):
        # Write sequential number
        f.write(f"{i}\n")
        # Write coordinates (longitude, latitude)
        f.write(f"{node_data.geometry.x}, {node_data.geometry.y}\n")





print(f"Original nodes: {len(nodes)}")
print(f"Filtered nodes (degree >= 2): {len(filtered_nodes)}")
print("Filtered data saved as venice_nodes_filtered.txt!")



################################
#############################  15/9/2025  START #################
print("15/9/2025 code started.....")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from networkx.algorithms.community import greedy_modularity_communities

# --------- Step 1: Load nodes ---------
nodes = {}
with open("venice_nodes_filtered.txt", "r") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        node_id = parts[0]
        lat, lon = float(parts[1]), float(parts[2])
        nodes[node_id] = (lon, lat)  # (x,y)

# --------- Step 2: Load edges ---------
edges = []
with open("venice_edges.txt", "r") as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        u, v = parts[0], parts[1]
        if u in nodes and v in nodes:
            edges.append((u, v))

# --------- Step 3: Build graph ---------
G = nx.Graph()
for n, (x, y) in nodes.items():
    G.add_node(n, pos=(x, y))
G.add_edges_from(edges)

# --------- Step 4: Detect communities ---------
communities = list(greedy_modularity_communities(G))
node_to_comm = {}
for i, comm in enumerate(communities):
    for node in comm:
        node_to_comm[node] = i

# --------- Step 5: Prepare data for classification ---------
node_ids = list(nodes.keys())
X = np.array([nodes[n] for n in node_ids])
y = np.array([node_to_comm[n] for n in node_ids])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --------- Step 6: Train KNN ---------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# --------- Step 7: Train SVM ---------
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# --------- Step 8: Visualize results ---------
def plot_results(X, y_pred, title):
    plt.figure(figsize=(8, 8))
    for lab in np.unique(y_pred):
        idx = y_pred == lab
        plt.scatter(X[idx, 0], X[idx, 1], s=10, label=f"Community {lab}")
    plt.title(title)
    plt.xlabel("Longitude (x)")
    plt.ylabel("Latitude (y)")
    plt.legend()
    plt.show()

plot_results(X_test, knn_pred, "KNN - Predicted Communities")
plot_results(X_test, svm_pred, "SVM - Predicted Communities")









print("VERSION 2 15/9/2025 has started.....")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from networkx.algorithms.community import greedy_modularity_communities
import seaborn as sns

print("Loading Venice network data for community classification...")

# --------- Step 1: Load nodes with error handling ---------
nodes = {}
try:
    print("Loading nodes from venice_nodes_filtered.txt...")
    with open("venice_nodes_filtered.txt", "r") as f:
        header = f.readline().strip()
        print(f"Header: {header}")
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
                
            # Try different separators
            parts = None
            if ',' in line:
                parts = line.split(',')
            elif '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()
            
            if len(parts) < 3:
                print(f"Warning: Line {line_num} has insufficient columns: {parts}")
                continue
                
            try:
                node_id = parts[0].strip('"').strip("'")  # Remove quotes if present
                # Handle different column orders
                if len(parts) >= 4:  # osmid, y, x, other_cols
                    lat, lon = float(parts[1]), float(parts[2])
                else:  # node_id, lat, lon
                    lat, lon = float(parts[1]), float(parts[2])
                
                nodes[node_id] = (lon, lat)  # (x,y) format
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num}: {line} - {e}")
                continue
    
    print(f"Successfully loaded {len(nodes)} nodes")

except FileNotFoundError:
    print("Error: venice_nodes_filtered.txt not found")
    print("Trying alternative file names...")
    
    # Try CSV format
    try:
        df = pd.read_csv("venice_nodes_filtered.csv")
        print("Found CSV file, extracting nodes...")
        
        # Handle different column names
        if 'osmid' in df.columns:
            node_col = 'osmid'
        else:
            node_col = df.columns[0]
            
        for _, row in df.iterrows():
            node_id = str(row[node_col])
            if 'x' in df.columns and 'y' in df.columns:
                lon, lat = row['x'], row['y']
            elif 'lon' in df.columns and 'lat' in df.columns:
                lon, lat = row['lon'], row['lat']
            else:
                print("Error: Could not find coordinate columns")
                exit()
            
            nodes[node_id] = (lon, lat)
        
        print(f"Successfully loaded {len(nodes)} nodes from CSV")
        
    except FileNotFoundError:
        print("Error: Could not find node file in any format")
        exit()

if len(nodes) == 0:
    print("Error: No nodes loaded successfully")
    exit()

# --------- Step 2: Load edges with error handling ---------
edges = []
edge_files = ["venice_edges.txt", "venice_edges_filtered.txt", "venice_edges_2core.txt"]

for edge_file in edge_files:
    try:
        print(f"Trying to load edges from {edge_file}...")
        with open(edge_file, "r") as f:
            header = f.readline().strip()
            print(f"Edge file header: {header}")
            
            for line_num, line in enumerate(f, start=2):
                line = line.strip()
                if not line:
                    continue
                
                # Try different separators
                if '\t' in line:
                    parts = line.split('\t')
                elif ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                
                if len(parts) < 2:
                    continue
                
                u, v = parts[0].strip('"').strip("'"), parts[1].strip('"').strip("'")
                
                # Only add edge if both nodes exist
                if u in nodes and v in nodes:
                    edges.append((u, v))
        
        print(f"Successfully loaded {len(edges)} edges from {edge_file}")
        break
        
    except FileNotFoundError:
        continue

if len(edges) == 0:
    print("Warning: No edges loaded. Creating a complete graph for demonstration.")
    # Create edges between nearby nodes for demonstration
    node_list = list(nodes.keys())
    coords = list(nodes.values())
    from scipy.spatial.distance import cdist
    
    distances = cdist(coords, coords)
    threshold = np.percentile(distances[distances > 0], 5)  # Connect to 5% nearest neighbors
    
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if i < j and distances[i, j] < threshold:
                edges.append((node1, node2))
    
    print(f"Created {len(edges)} edges based on proximity")

# --------- Step 3: Build graph ---------
print("Building NetworkX graph...")
G = nx.Graph()

# Add nodes with positions
for node_id, (x, y) in nodes.items():
    G.add_node(node_id, pos=(x, y))

# Add edges
G.add_edges_from(edges)

# Remove isolated nodes
isolated = list(nx.isolates(G))
if isolated:
    G.remove_nodes_from(isolated)
    print(f"Removed {len(isolated)} isolated nodes")

print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

if G.number_of_nodes() < 10:
    print("Error: Graph too small for community detection")
    exit()

# --------- Step 4: Detect communities ---------
print("Detecting communities using greedy modularity...")

try:
    communities = list(greedy_modularity_communities(G))
    print(f"Found {len(communities)} communities")
    
    # Display community sizes
    comm_sizes = [len(comm) for comm in communities]
    print("Community sizes:", sorted(comm_sizes, reverse=True))
    
    # Create node to community mapping
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i
    
    # Calculate modularity
    modularity = nx.algorithms.community.modularity(G, communities)
    print(f"Modularity score: {modularity:.4f}")

except Exception as e:
    print(f"Error in community detection: {e}")
    exit()

# Filter out very small communities (less than 3 nodes)
min_community_size = 3
large_communities = [comm for comm in communities if len(comm) >= min_community_size]

if len(large_communities) < 2:
    print("Error: Not enough large communities for classification")
    exit()

print(f"Using {len(large_communities)} communities with size >= {min_community_size}")

# Rebuild community mapping with only large communities
node_to_comm = {}
for i, comm in enumerate(large_communities):
    for node in comm:
        node_to_comm[node] = i

# --------- Step 5: Prepare data for classification ---------
print("Preparing data for classification...")

# Get nodes that belong to large communities
valid_nodes = [n for n in G.nodes() if n in node_to_comm]
print(f"Using {len(valid_nodes)} nodes for classification")

X = np.array([nodes[n] for n in valid_nodes])
y = np.array([node_to_comm[n] for n in valid_nodes])

print(f"Feature matrix shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print("Class distribution:", {i: np.sum(y == i) for i in np.unique(y)})

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# --------- Step 6: Train classifiers ---------
print("\nTraining classifiers...")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

# --------- Step 7: Evaluate results ---------
print("\n" + "="*50)
print("CLASSIFICATION RESULTS")
print("="*50)

print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

print(f"\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))

print(f"\nSVM Classification Report:")
print(classification_report(y_test, svm_pred))

# --------- Step 8: Comprehensive visualization ---------
print("\nCreating visualizations...")

# Create subplot layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Venice Network Community Classification Analysis', fontsize=16)

# 1. True communities (all nodes)
ax1 = axes[0, 0]
X_all_scaled = scaler.transform(np.array([nodes[n] for n in valid_nodes]))
y_all = np.array([node_to_comm[n] for n in valid_nodes])

colors = plt.cm.tab10(np.linspace(0, 1, len(large_communities)))
for i, comm in enumerate(large_communities):
    comm_nodes = [n for n in comm if n in valid_nodes]
    comm_coords = [nodes[n] for n in comm_nodes]
    if comm_coords:
        coords_array = np.array(comm_coords)
        ax1.scatter(coords_array[:, 0], coords_array[:, 1], 
                   c=[colors[i]], label=f'Community {i} ({len(comm_nodes)})', 
                   s=15, alpha=0.7)

ax1.set_title("True Communities (Network-based)")
ax1.set_xlabel("Longitude (x)")
ax1.set_ylabel("Latitude (y)")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Test data - True labels
ax2 = axes[0, 1]
X_test_orig = scaler.inverse_transform(X_test)
scatter2 = ax2.scatter(X_test_orig[:, 0], X_test_orig[:, 1], c=y_test, 
                      cmap='tab10', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
ax2.set_title("Test Nodes - True Communities")
ax2.set_xlabel("Longitude (x)")
ax2.set_ylabel("Latitude (y)")
plt.colorbar(scatter2, ax=ax2)

# 3. Test data - KNN predictions
ax3 = axes[0, 2]
scatter3 = ax3.scatter(X_test_orig[:, 0], X_test_orig[:, 1], c=knn_pred, 
                      cmap='tab10', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
ax3.set_title(f"Test Nodes - KNN Predictions (Acc: {knn_accuracy:.3f})")
ax3.set_xlabel("Longitude (x)")
ax3.set_ylabel("Latitude (y)")
plt.colorbar(scatter3, ax=ax3)

# 4. Test data - SVM predictions
ax4 = axes[1, 0]
scatter4 = ax4.scatter(X_test_orig[:, 0], X_test_orig[:, 1], c=svm_pred, 
                      cmap='tab10', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
ax4.set_title(f"Test Nodes - SVM Predictions (Acc: {svm_accuracy:.3f})")
ax4.set_xlabel("Longitude (x)")
ax4.set_ylabel("Latitude (y)")
plt.colorbar(scatter4, ax=ax4)

# 5. KNN Confusion Matrix
ax5 = axes[1, 1]
cm_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_knn, fmt='d', cmap='Blues', ax=ax5, square=True, cbar=True)
ax5.set_title("KNN Confusion Matrix")
ax5.set_xlabel("Predicted Community")
ax5.set_ylabel("True Community")

ax5.set_xticks(np.arange(cm_knn.shape[1]) + 0.5)
ax5.set_yticks(np.arange(cm_knn.shape[0]) + 0.5)
ax5.set_xticklabels(range(cm_knn.shape[1]), rotation=90)
ax5.set_yticklabels(range(cm_knn.shape[0]), rotation=0)


# 6. SVM Confusion Matrix
ax6 = axes[1, 2]
cm_svm = confusion_matrix(y_test, svm_pred)

sns.heatmap(cm_svm, fmt='d', cmap='Blues', ax=ax6, square=True, cbar=True)
ax6.set_title("SVM Confusion Matrix")
ax6.set_xlabel("Predicted Community")
ax6.set_ylabel("True Community")

ax6.set_xticks(np.arange(cm_svm.shape[1]) + 0.5)
ax6.set_yticks(np.arange(cm_svm.shape[0]) + 0.5)
ax6.set_xticklabels(range(cm_svm.shape[1]), rotation=90)
ax6.set_yticklabels(range(cm_svm.shape[0]), rotation=0)

plt.tight_layout()
plt.show()

# --------- Step 9: Community analysis ---------
print("\nCommunity Analysis:")
print("-" * 40)

for i, comm in enumerate(large_communities):
    comm_nodes = [n for n in comm if n in valid_nodes]
    if comm_nodes:
        comm_coords = np.array([nodes[n] for n in comm_nodes])
        
        print(f"Community {i}:")
        print(f"  Size: {len(comm_nodes)} nodes")
        print(f"  Geographic bounds:")
        print(f"    Longitude: {comm_coords[:, 0].min():.6f} to {comm_coords[:, 0].max():.6f}")
        print(f"    Latitude:  {comm_coords[:, 1].min():.6f} to {comm_coords[:, 1].max():.6f}")
        print(f"  Geographic spread:")
        print(f"    Longitude range: {comm_coords[:, 0].max() - comm_coords[:, 0].min():.6f}")
        print(f"    Latitude range:  {comm_coords[:, 1].max() - comm_coords[:, 1].min():.6f}")
        print()

# --------- Step 10: Save results ---------
print("Saving results...")

results_df = pd.DataFrame({
    'node_id': valid_nodes,
    'longitude': [nodes[n][0] for n in valid_nodes],
    'latitude': [nodes[n][1] for n in valid_nodes], 
    'true_community': [node_to_comm[n] for n in valid_nodes]
})

# Add predictions for all nodes
all_predictions_knn = knn.predict(scaler.transform(np.array([nodes[n] for n in valid_nodes])))
all_predictions_svm = svm.predict(scaler.transform(np.array([nodes[n] for n in valid_nodes])))

results_df['knn_prediction'] = all_predictions_knn
results_df['svm_prediction'] = all_predictions_svm

results_df.to_csv("venice_community_classification_results.csv", index=False)

print("Results saved to venice_community_classification_results.csv")
print(f"\nCommunity classification complete!")
print(f"Modularity: {modularity:.4f}")
print(f"Best classifier: {'KNN' if knn_accuracy > svm_accuracy else 'SVM'} "
      f"(Accuracy: {max(knn_accuracy, svm_accuracy):.4f})")

# --------- Step 11: Network visualization ---------
print("\nCreating network visualization...")

plt.figure(figsize=(14, 10))

# Get positions
pos = {n: nodes[n] for n in G.nodes() if n in valid_nodes}

# Draw network with community colors
for i, comm in enumerate(large_communities):
    comm_nodes = [n for n in comm if n in G.nodes() and n in valid_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes, 
                          node_color=[colors[i]], node_size=20, 
                          alpha=0.7, label=f'Community {i}')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color='gray')

#plt.figure(figsize=(14, 10))
plt.title(f'Venice Network Communities (Modularity: {modularity:.3f})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
#plt.axis('equal')
plt.tight_layout()
plt.show()

print("Analysis complete!")
#############################  15/9/2025  END #################


###################### KNNNNNNNNN AND SVMMMMMMMMMMMMMMMMMMMMMMMMM
print("KNN and SVM code started..........")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- Load the nodes from CSV (not TXT) ---
print("Loading data...")
try:
    # Try to load as CSV first (most likely format from OSMnx)
    df = pd.read_csv("venice_nodes_filtered.csv")
    print(f"Loaded {len(df)} nodes from CSV file")
except FileNotFoundError:
    try:
        # If CSV doesn't exist, try the TXT file
        df = pd.read_csv("venice_nodes_filtered.txt")
        print(f"Loaded {len(df)} nodes from TXT file")
    except FileNotFoundError:
        print("Error: Could not find venice_nodes_filtered.csv or venice_nodes_filtered.txt")
        print("Make sure you've run the Venice network extraction code first")
        exit()

# --- Check and prepare data ---
print("\nData columns:", df.columns.tolist())
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Handle different possible column names from OSMnx output
if 'x' in df.columns and 'y' in df.columns:
    # x and y columns already exist, no need to extract from geometry
    print("Using existing x and y columns")
elif 'geometry' in df.columns:
    try:
        # Try to extract x, y from geometry objects
        if hasattr(df.geometry.iloc[0], 'x'):
            df['x'] = df.geometry.x
            df['y'] = df.geometry.y
        else:
            # If geometry is stored as string, parse it
            from shapely import wkt
            df['geometry'] = df['geometry'].apply(wkt.loads)
            df['x'] = df.geometry.x
            df['y'] = df.geometry.y
    except Exception as e:
        print(f"Error processing geometry: {e}")
        # If geometry processing fails, check for other coordinate columns
        if 'lon' in df.columns and 'lat' in df.columns:
            df['x'] = df['lon']
            df['y'] = df['lat']
        else:
            print("Error: Could not find or extract coordinate information")
            exit()
elif 'lon' in df.columns and 'lat' in df.columns:
    df['x'] = df['lon']
    df['y'] = df['lat']

# Check if we have the required columns
required_cols = ['x', 'y']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\nError: Missing required columns: {missing_cols}")
    exit()

# --- Select features ---
print("\nPreparing features...")
X = df[['x', 'y']].values
print(f"Feature matrix shape: {X.shape}")
print(f"Feature ranges:")
print(f"  x: {X[:, 0].min():.6f} to {X[:, 0].max():.6f}")
print(f"  y: {X[:, 1].min():.6f} to {X[:, 1].max():.6f}")

# --- STEP 1: Cluster to create labels ---
print("\nPerforming K-means clustering...")
n_clusters = 6   # you can change this
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Scale features for clustering (important for K-means)
scaler_cluster = StandardScaler()
X_scaled_cluster = scaler_cluster.fit_transform(X)
cluster_labels = kmeans.fit_predict(X_scaled_cluster)

df['label'] = cluster_labels
y = df['label'].values

print(f"Created {n_clusters} clusters")
print("Cluster distribution:")
for i in range(n_clusters):
    count = np.sum(y == i)
    print(f"  Cluster {i}: {count} nodes ({count/len(y)*100:.1f}%)")

# --- STEP 2: Train/test split ---
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# --- STEP 3: Scale features for classification ---
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- STEP 4: Train kNN ---
print("\nTraining k-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# --- STEP 5: Train SVM ---
print("Training Support Vector Machine...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# --- STEP 6: Evaluate ---
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print(f"k-NN Accuracy: {knn_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

print(f"\nk-NN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print(f"\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# --- STEP 7: Visualization ---
print("\nCreating visualizations...")

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Venice Node Geographic Classification Analysis', fontsize=16)

# 1. All nodes colored by cluster
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df['x'], df['y'], c=df['label'], cmap='tab10', s=15, alpha=0.7)
ax1.set_title("All Nodes - K-means Clusters")
ax1.set_xlabel("Longitude (x)")
ax1.set_ylabel("Latitude (y)")
plt.colorbar(scatter1, ax=ax1)

# 2. Nodes colored by density (optional - geographic density)
ax2 = axes[0, 1]
# Calculate local density for visualization
from scipy.spatial import cKDTree
tree = cKDTree(df[['x', 'y']].values)
densities = []
radius = 0.001  # Small radius for local density
for point in df[['x', 'y']].values:
    nearby = tree.query_ball_point(point, r=radius)
    densities.append(len(nearby))

df['local_density'] = densities
scatter2 = ax2.scatter(df['x'], df['y'], c=df['local_density'], cmap='viridis', s=15, alpha=0.7)
ax2.set_title("Nodes by Local Density")
ax2.set_xlabel("Longitude (x)")
ax2.set_ylabel("Latitude (y)")
plt.colorbar(scatter2, ax=ax2)

# 3. Test nodes - k-NN predictions
ax3 = axes[0, 2]
test_df = pd.DataFrame(X_test, columns=['x', 'y'])
test_df['true_label'] = y_test
test_df['pred_knn'] = y_pred_knn
test_df['pred_svm'] = y_pred_svm

scatter3 = ax3.scatter(test_df['x'], test_df['y'], c=test_df['pred_knn'], 
                      cmap='tab10', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
ax3.set_title("Test Nodes - k-NN Predictions")
ax3.set_xlabel("Longitude (x)")
ax3.set_ylabel("Latitude (y)")
plt.colorbar(scatter3, ax=ax3)

# 4. Test nodes - SVM predictions
ax4 = axes[1, 0]
scatter4 = ax4.scatter(test_df['x'], test_df['y'], c=test_df['pred_svm'], 
                      cmap='tab10', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
ax4.set_title("Test Nodes - SVM Predictions")
ax4.set_xlabel("Longitude (x)")
ax4.set_ylabel("Latitude (y)")
plt.colorbar(scatter4, ax=ax4)

# 5. Confusion matrix for kNN
ax5 = axes[1, 1]
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax5)
ax5.set_title("k-NN Confusion Matrix")
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")

# 6. Confusion matrix for SVM
ax6 = axes[1, 2]
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax6)
ax6.set_title("SVM Confusion Matrix")
ax6.set_xlabel("Predicted")
ax6.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# --- STEP 8: Geographic cluster analysis ---
print("\nGeographic Cluster Analysis:")
print("-" * 40)

# Analyze cluster centers
cluster_centers = scaler_cluster.inverse_transform(kmeans.cluster_centers_)
feature_names = ['x', 'y']

print("Cluster Centers (Geographic):")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}:")
    for j, feature in enumerate(feature_names):
        print(f"  {feature}: {center[j]:.6f}")
    
    # Calculate cluster statistics
    cluster_points = df[df['label'] == i]
    print(f"  Size: {len(cluster_points)} nodes")
    print(f"  Geographic spread:")
    print(f"    x range: {cluster_points['x'].max() - cluster_points['x'].min():.6f}")
    print(f"    y range: {cluster_points['y'].max() - cluster_points['y'].min():.6f}")
    print()

# --- STEP 9: Additional geographic analysis ---
print("Geographic Compactness Analysis:")
print("-" * 40)

from scipy.spatial.distance import cdist

for i in range(n_clusters):
    cluster_points = df[df['label'] == i][['x', 'y']].values
    if len(cluster_points) > 1:
        # Calculate average intra-cluster distance
        distances = cdist(cluster_points, cluster_points)
        avg_distance = np.mean(distances[distances > 0])
        max_distance = np.max(distances)
        
        print(f"Cluster {i}:")
        print(f"  Average intra-cluster distance: {avg_distance:.6f}")
        print(f"  Maximum intra-cluster distance: {max_distance:.6f}")
        print(f"  Compactness ratio: {avg_distance/max_distance:.3f}")
        print()

# --- STEP 10: Save results ---
print("Saving results...")

# Save predictions
results_df = df.copy()
results_df['cluster_label'] = cluster_labels

# Add predictions for all data (not just test)
all_predictions_knn = knn.predict(scaler.transform(X))
all_predictions_svm = svm.predict(scaler.transform(X))

results_df['knn_prediction'] = all_predictions_knn
results_df['svm_prediction'] = all_predictions_svm

results_df.to_csv("venice_geographic_classification_results.csv", index=False)

print("Results saved to venice_geographic_classification_results.csv")
print("\nGeographic classification complete!")

# --- STEP 11: Create a summary map ---
print("\nCreating summary geographic visualization...")

plt.figure(figsize=(12, 10))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for i in range(n_clusters):
    cluster_points = df[df['label'] == i]
    plt.scatter(cluster_points['x'], cluster_points['y'], 
               c=colors[i % len(colors)], label=f'Cluster {i} ({len(cluster_points)} nodes)',
               s=20, alpha=0.7)

plt.xlabel('Longitude (x)')
plt.ylabel('Latitude (y)')
plt.title('Venice Network Nodes - Geographic Clusters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Analysis complete! Check the generated visualizations and CSV file for results.")
###################### KNNNNNNNNN AND SVMMMMMMMMMMMMMMMMMMMMMMMMM ENDDDDDDDDDDDd






# Community Detection and Visualization for Venice Network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import osmnx as ox
from matplotlib.colors import ListedColormap
import seaborn as sns

# -----------------------------
# Community Detection using K-Means Clustering
# -----------------------------

def perform_community_detection(graph, n_communities=5, method='kmeans'):
    """
    Perform community detection on the graph using different methods
    
    Parameters:
    - graph: NetworkX graph
    - n_communities: number of communities to detect
    - method: 'kmeans', 'networkx', or 'hybrid'
    """
    
    # Get node positions for spatial clustering
    nodes_gdf, _ = ox.graph_to_gdfs(graph)
    
    if method == 'kmeans':
        # Method 1: Spatial K-Means clustering based on geographical coordinates
        coords = np.array([[node_data['y'], node_data['x']] for node_id, node_data in graph.nodes(data=True)])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
        community_labels = kmeans.fit_predict(coords)
        
        # Create community dictionary
        communities = {}
        for i, node_id in enumerate(graph.nodes()):
            communities[node_id] = community_labels[i]
            
    elif method == 'networkx':
        # Method 2: NetworkX community detection (Louvain algorithm)
        try:
            import networkx.algorithms.community as nx_comm
            communities_sets = nx_comm.greedy_modularity_communities(graph.to_undirected())
            
            communities = {}
            for i, community_set in enumerate(communities_sets[:n_communities]):
                for node in community_set:
                    communities[node] = i
        except ImportError:
            print("NetworkX community module not available, falling back to k-means")
            return perform_community_detection(graph, n_communities, 'kmeans')
            
    elif method == 'hybrid':
        # Method 3: True Hybrid Community Detection
        # Combines topological community detection with spatial constraints
        
        # Step 1: Convert multigraph to simple graph for community detection
        simple_graph = nx.Graph(graph)
        print(f"Converting multigraph ({len(graph.edges())} edges) to simple graph ({len(simple_graph.edges())} edges)")
        
        # Step 2: Initial community detection using pure network topology
        try:
            import networkx.algorithms.community as nx_comm
            print("Running initial topological community detection...")
            
            # Use Louvain-like algorithm (greedy modularity)
            initial_communities = list(nx_comm.greedy_modularity_communities(simple_graph))
            print(f"Found {len(initial_communities)} initial topological communities")
            
            # Convert to node->community mapping
            topo_communities = {}
            for i, community_set in enumerate(initial_communities):
                for node in community_set:
                    topo_communities[node] = i
                    
        except ImportError:
            print("NetworkX community module not available, using label propagation")
            # Fallback: Label Propagation Algorithm
            topo_communities = nx.algorithms.community.label_propagation_communities(simple_graph)
            topo_communities = {node: i for i, community_set in enumerate(topo_communities) 
                              for node in community_set}
        
        # Step 3: Spatial refinement of topological communities
        print("Applying spatial refinement...")
        
        # Get node coordinates
        node_coords = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_coords[node_id] = (node_data['y'], node_data['x'])
        
        # Calculate spatial distances within each topological community
        refined_communities = {}
        community_id_counter = 0
        
        for topo_comm_id in set(topo_communities.values()):
            # Get nodes in this topological community
            nodes_in_topo_comm = [node for node, comm in topo_communities.items() if comm == topo_comm_id]
            
            if len(nodes_in_topo_comm) <= 2:
                # Small communities stay as is
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
                continue
            
            # Extract coordinates for this community
            comm_coords = np.array([node_coords[node] for node in nodes_in_topo_comm])
            
            # Determine number of spatial subclusters
            # Use elbow method or maximum of 3 subclusters per topological community
            max_subclusters = min(3, max(2, len(nodes_in_topo_comm) // 10))
            
            if len(nodes_in_topo_comm) > 5:
                # Apply k-means to spatially subdivide large topological communities
                kmeans = KMeans(n_clusters=max_subclusters, random_state=42, n_init=10)
                spatial_labels = kmeans.fit_predict(comm_coords)
                
                # Assign refined community IDs
                for i, node in enumerate(nodes_in_topo_comm):
                    refined_communities[node] = community_id_counter + spatial_labels[i]
                community_id_counter += max_subclusters
            else:
                # Keep small topological communities intact
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
        
        # Step 4: Post-processing to ensure we have the desired number of communities
        current_communities = len(set(refined_communities.values()))
        print(f"Refined communities: {current_communities}")
        
        if current_communities > n_communities:
            # Merge smallest communities
            community_sizes = {}
            for comm_id in set(refined_communities.values()):
                community_sizes[comm_id] = sum(1 for c in refined_communities.values() if c == comm_id)
            
            # Sort communities by size
            sorted_comms = sorted(community_sizes.items(), key=lambda x: x[1])
            communities_to_merge = sorted_comms[:current_communities - n_communities]
            
            # Merge smallest communities with their spatially nearest larger community
            for small_comm_id, _ in communities_to_merge:
                nodes_in_small = [node for node, comm in refined_communities.items() if comm == small_comm_id]
                
                if nodes_in_small:
                    # Find center of small community
                    small_coords = np.array([node_coords[node] for node in nodes_in_small])
                    small_center = np.mean(small_coords, axis=0)
                    
                    # Find nearest larger community
                    min_distance = float('inf')
                    target_comm = None
                    
                    for other_comm_id in set(refined_communities.values()):
                        if other_comm_id != small_comm_id and community_sizes[other_comm_id] > community_sizes[small_comm_id]:
                            other_nodes = [node for node, comm in refined_communities.items() if comm == other_comm_id]
                            other_coords = np.array([node_coords[node] for node in other_nodes])
                            other_center = np.mean(other_coords, axis=0)
                            
                            distance = np.linalg.norm(small_center - other_center)
                            if distance < min_distance:
                                min_distance = distance
                                target_comm = other_comm_id
                    
                    # Merge communities
                    if target_comm is not None:
                        for node in nodes_in_small:
                            refined_communities[node] = target_comm
        
        # Relabel communities to be consecutive integers starting from 0
        unique_comms = sorted(set(refined_communities.values()))
        comm_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_comms)}
        communities = {node: comm_mapping[comm_id] for node, comm_id in refined_communities.items()}
        
        print(f"Final hybrid communities: {len(set(communities.values()))}")
    
    return communities

def plot_communities(graph, communities, title="Network Communities", figsize=(15, 10)):
    """
    Plot the graph with nodes colored by community
    """
    # Create color palette
    n_communities = len(set(communities.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, n_communities))
    
    # Create node colors list
    node_colors = [colors[communities[node]] for node in graph.nodes()]
    
    # Plot the graph
    fig, ax = ox.plot_graph(
        graph,
        node_size=15,
        node_color=node_colors,
        edge_color="gray",
        bgcolor="black",
        figsize=figsize,
        show=False,
        close=False
    )
    
    ax.set_title(title, color="white", fontsize=16, pad=20)
    
    # Add legend
    legend_elements = []
    for i in range(n_communities):
        legend_elements.append(plt.scatter([], [], c=[colors[i]], s=50, label=f'Community {i+1}'))
    
    ax.legend(handles=legend_elements, loc='upper right', 
              facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def analyze_communities(graph, communities):
    """
    Analyze the detected communities
    """
    print("Community Analysis:")
    print("-" * 50)
    
    community_stats = {}
    for community_id in set(communities.values()):
        nodes_in_community = [node for node, comm in communities.items() if comm == community_id]
        community_stats[community_id] = {
            'size': len(nodes_in_community),
            'avg_degree': np.mean([graph.degree[node] for node in nodes_in_community])
        }
    
    for comm_id, stats in community_stats.items():
        print(f"Community {comm_id + 1}: {stats['size']} nodes, avg degree: {stats['avg_degree']:.2f}")
    
    # Calculate modularity if possible
    try:
        import networkx.algorithms.community as nx_comm
        partition = [set([node for node, comm in communities.items() if comm == comm_id]) 
                    for comm_id in set(communities.values())]
        modularity = nx_comm.modularity(graph.to_undirected(), partition)
        print(f"\nModularity Score: {modularity:.3f}")
    except:
        print("\nModularity calculation not available")

# -----------------------------
# Apply Community Detection to Venice Network
# -----------------------------

print("Performing Community Detection on Venice Network...")
print("=" * 60)

# Apply different methods
methods = ['kmeans', 'hybrid']
n_communities = 3  # You can adjust this number

for method in methods:
    print(f"\nMethod: {method.upper()}")
    try:
        communities = perform_community_detection(G_core, n_communities=n_communities, method=method)
        
        # Plot results
        plot_communities(G_core, communities, 
                        title=f"Venice Network - {method.upper()} Communities (k={n_communities})")
        
        # Analyze communities
        analyze_communities(G_core, communities)
        
        # Save community data
        filename = f"venice_communities_{method}.txt"
        with open(filename, "w") as f:
            f.write("Node_ID\tCommunity\n")
            for node_id, community_id in communities.items():
                f.write(f"{node_id}\t{community_id}\n")
        print(f"Community data saved to {filename}")
        
    except Exception as e:
        print(f"Error with {method}: {e}")

print("\n" + "=" * 60)
print("Community detection completed!")

# Optional: Compare different numbers of communities
print("\nComparing different numbers of communities (K-Means method):")
#k_values = [3, 4, 5, 6, 7, 8]
k_values = [2]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

#for i, k in enumerate(k_values):
#    communities = perform_community_detection(G_core, n_communities=k, method='kmeans')
    
    # Create subplot
#    ax = axes[i]
    
    # Get node positions
#    pos = {node: (data['x'], data['y']) for node, data in G_core.nodes(data=True)}
    
    # Create colors
#    colors = plt.cm.Set3(np.linspace(0, 1, k))
#    node_colors = [colors[communities[node]] for node in G_core.nodes()]
    
    # Draw network
#    nx.draw_networkx(G_core, pos, ax=ax, node_color=node_colors, 
#                    node_size=10, edge_color='gray', with_labels=False)
#    ax.set_title(f'K = {k} Communities', fontsize=12)
#    ax.set_facecolor('black')

#plt.tight_layout()
#plt.suptitle('Venice Network: Different Numbers of Communities', 
#             fontsize=16, y=0.98, color='white')
#plt.show()





######################################
# Community Detection and Visualization for Venice Network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import osmnx as ox
from matplotlib.colors import ListedColormap
import seaborn as sns

# -----------------------------
# Community Detection using K-Means Clustering
# -----------------------------

def perform_community_detection(graph, n_communities=5, method='kmeans'):
    """
    Perform community detection on the graph using different methods
    
    Parameters:
    - graph: NetworkX graph
    - n_communities: number of communities to detect
    - method: 'kmeans', 'networkx', or 'hybrid'
    """
    
    # Get node positions for spatial clustering
    nodes_gdf, _ = ox.graph_to_gdfs(graph)
    
    if method == 'kmeans':
        # Method 1: Spatial K-Means clustering based on geographical coordinates
        coords = np.array([[node_data['y'], node_data['x']] for node_id, node_data in graph.nodes(data=True)])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
        community_labels = kmeans.fit_predict(coords)
        
        # Create community dictionary
        communities = {}
        for i, node_id in enumerate(graph.nodes()):
            communities[node_id] = community_labels[i]
            
    elif method == 'networkx':
        # Method 2: NetworkX community detection (Louvain algorithm)
        try:
            import networkx.algorithms.community as nx_comm
            communities_sets = nx_comm.greedy_modularity_communities(graph.to_undirected())
            
            communities = {}
            for i, community_set in enumerate(communities_sets[:n_communities]):
                for node in community_set:
                    communities[node] = i
        except ImportError:
            print("NetworkX community module not available, falling back to k-means")
            return perform_community_detection(graph, n_communities, 'kmeans')
    
    elif method == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split

        # Get coordinates of all nodes
        coords = np.array([[data['y'], data['x']] for _, data in graph.nodes(data=True)])
        node_ids = list(graph.nodes())

        # Step 1: create some pseudo-labels using KMeans on a small sample of nodes
        n_seed = max(50, len(coords) // 10)  # about 10% of nodes or 50 minimum
        sample_idx = np.random.choice(len(coords), size=n_seed, replace=False)
        sample_coords = coords[sample_idx]

        kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
        seed_labels = kmeans.fit_predict(sample_coords)

        # Step 2: train kNN classifier on seed nodes
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(sample_coords, seed_labels)

        # Step 3: predict communities for all nodes
        predicted_labels = knn.predict(coords)

        # Build communities dict
        communities = {node_ids[i]: int(predicted_labels[i]) for i in range(len(node_ids))}
   
    elif method == 'hybrid':
        # Method 3: True Hybrid Community Detection
        # Combines topological community detection with spatial constraints
        
        # Step 1: Convert multigraph to simple graph for community detection
        simple_graph = nx.Graph(graph)
        print(f"Converting multigraph ({len(graph.edges())} edges) to simple graph ({len(simple_graph.edges())} edges)")
        
        # Step 2: Initial community detection using pure network topology
        try:
            import networkx.algorithms.community as nx_comm
            print("Running initial topological community detection...")
            
            # Use Louvain-like algorithm (greedy modularity)
            initial_communities = list(nx_comm.greedy_modularity_communities(simple_graph))
            print(f"Found {len(initial_communities)} initial topological communities")
            
            # Convert to node->community mapping
            topo_communities = {}
            for i, community_set in enumerate(initial_communities):
                for node in community_set:
                    topo_communities[node] = i
                    
        except ImportError:
            print("NetworkX community module not available, using label propagation")
            # Fallback: Label Propagation Algorithm
            topo_communities = nx.algorithms.community.label_propagation_communities(simple_graph)
            topo_communities = {node: i for i, community_set in enumerate(topo_communities) 
                              for node in community_set}
        
        # Step 3: Spatial refinement of topological communities
        print("Applying spatial refinement...")
        
        # Get node coordinates
        node_coords = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_coords[node_id] = (node_data['y'], node_data['x'])
        
        # Calculate spatial distances within each topological community
        refined_communities = {}
        community_id_counter = 0
        
        for topo_comm_id in set(topo_communities.values()):
            # Get nodes in this topological community
            nodes_in_topo_comm = [node for node, comm in topo_communities.items() if comm == topo_comm_id]
            
            if len(nodes_in_topo_comm) <= 2:
                # Small communities stay as is
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
                continue
            
            # Extract coordinates for this community
            comm_coords = np.array([node_coords[node] for node in nodes_in_topo_comm])
            
            # Determine number of spatial subclusters
            # Use elbow method or maximum of 3 subclusters per topological community
            max_subclusters = min(3, max(2, len(nodes_in_topo_comm) // 10))
            
            if len(nodes_in_topo_comm) > 5:
                # Apply k-means to spatially subdivide large topological communities
                kmeans = KMeans(n_clusters=max_subclusters, random_state=42, n_init=10)
                spatial_labels = kmeans.fit_predict(comm_coords)
                
                # Assign refined community IDs
                for i, node in enumerate(nodes_in_topo_comm):
                    refined_communities[node] = community_id_counter + spatial_labels[i]
                community_id_counter += max_subclusters
            else:
                # Keep small topological communities intact
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
        
        # Step 4: Post-processing to ensure we have the desired number of communities
        current_communities = len(set(refined_communities.values()))
        print(f"Refined communities: {current_communities}")
        
        if current_communities > n_communities:
            # Merge smallest communities
            community_sizes = {}
            for comm_id in set(refined_communities.values()):
                community_sizes[comm_id] = sum(1 for c in refined_communities.values() if c == comm_id)
            
            # Sort communities by size
            sorted_comms = sorted(community_sizes.items(), key=lambda x: x[1])
            communities_to_merge = sorted_comms[:current_communities - n_communities]
            
            # Merge smallest communities with their spatially nearest larger community
            for small_comm_id, _ in communities_to_merge:
                nodes_in_small = [node for node, comm in refined_communities.items() if comm == small_comm_id]
                
                if nodes_in_small:
                    # Find center of small community
                    small_coords = np.array([node_coords[node] for node in nodes_in_small])
                    small_center = np.mean(small_coords, axis=0)
                    
                    # Find nearest larger community
                    min_distance = float('inf')
                    target_comm = None
                    
                    for other_comm_id in set(refined_communities.values()):
                        if other_comm_id != small_comm_id and community_sizes[other_comm_id] > community_sizes[small_comm_id]:
                            other_nodes = [node for node, comm in refined_communities.items() if comm == other_comm_id]
                            other_coords = np.array([node_coords[node] for node in other_nodes])
                            other_center = np.mean(other_coords, axis=0)
                            
                            distance = np.linalg.norm(small_center - other_center)
                            if distance < min_distance:
                                min_distance = distance
                                target_comm = other_comm_id
                    
                    # Merge communities
                    if target_comm is not None:
                        for node in nodes_in_small:
                            refined_communities[node] = target_comm
        
        # Relabel communities to be consecutive integers starting from 0
        unique_comms = sorted(set(refined_communities.values()))
        comm_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_comms)}
        communities = {node: comm_mapping[comm_id] for node, comm_id in refined_communities.items()}
        
        print(f"Final hybrid communities: {len(set(communities.values()))}")
    
    return communities

def plot_communities(graph, communities, title="Network Communities", figsize=(15, 10)):
    """
    Plot the graph with nodes colored by community
    """
    # Create color palette
    n_communities = len(set(communities.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, n_communities))
    
    # Create node colors list
    node_colors = [colors[communities[node]] for node in graph.nodes()]
    
    # Plot the graph
    fig, ax = ox.plot_graph(
        graph,
        node_size=15,
        node_color=node_colors,
        edge_color="gray",
        bgcolor="black",
        figsize=figsize,
        show=False,
        close=False
    )
    
    ax.set_title(title, color="white", fontsize=16, pad=20)
    
    # Add legend
    legend_elements = []
    for i in range(n_communities):
        legend_elements.append(plt.scatter([], [], c=[colors[i]], s=50, label=f'Community {i+1}'))
    
    ax.legend(handles=legend_elements, loc='upper right', 
              facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def analyze_communities(graph, communities):
    """
    Analyze the detected communities
    """
    print("Community Analysis:")
    print("-" * 50)
    
    community_stats = {}
    for community_id in set(communities.values()):
        nodes_in_community = [node for node, comm in communities.items() if comm == community_id]
        community_stats[community_id] = {
            'size': len(nodes_in_community),
            'avg_degree': np.mean([graph.degree[node] for node in nodes_in_community])
        }
    
    for comm_id, stats in community_stats.items():
        print(f"Community {comm_id + 1}: {stats['size']} nodes, avg degree: {stats['avg_degree']:.2f}")
    
    # Calculate modularity if possible
    try:
        import networkx.algorithms.community as nx_comm
        partition = [set([node for node, comm in communities.items() if comm == comm_id]) 
                    for comm_id in set(communities.values())]
        modularity = nx_comm.modularity(graph.to_undirected(), partition)
        print(f"\nModularity Score: {modularity:.3f}")
    except:
        print("\nModularity calculation not available")

# -----------------------------
# Apply Community Detection to Venice Network
# -----------------------------

print("Performing Community Detection on Venice Network...")
print("=" * 60)

# Apply different methods
methods = ['kmeans', 'knn', 'hybrid']
n_communities = 6  # You can adjust this number

for method in methods:
    print(f"\nMethod: {method.upper()}")
    try:
        communities = perform_community_detection(G_core, n_communities=n_communities, method=method)
        
        # Plot results
        plot_communities(G_core, communities, 
                        title=f"Venice Network - {method.upper()} Communities (k={n_communities})")
        
        # Analyze communities
        analyze_communities(G_core, communities)
        
        # Save community data
        filename = f"venice_communities_{method}.txt"
        with open(filename, "w") as f:
            f.write("Node_ID\tCommunity\n")
            for node_id, community_id in communities.items():
                f.write(f"{node_id}\t{community_id}\n")
        print(f"Community data saved to {filename}")
        
    except Exception as e:
        print(f"Error with {method}: {e}")

print("\n" + "=" * 60)
print("Community detection completed!")






############################
###########################################crowded community detection and visualization
# Community Detection and Visualization for Venice Network
print("=" * 60)
print("Performing Crowded Community Detection on Venice Network...")
print("=" * 60)
print("=" * 60)
print("=" * 60)
print("=" * 60)
print("=" * 60)
# Community Detection and Visualization for Venice Network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import osmnx as ox
from matplotlib.colors import ListedColormap
import seaborn as sns

# -----------------------------
# Community Detection using K-Means Clustering
# -----------------------------

def perform_community_detection(graph, n_communities=5, method='kmeans'):
    """
    Perform community detection on the graph using different methods
    
    Parameters:
    - graph: NetworkX graph
    - n_communities: number of communities to detect
    - method: 'kmeans', 'networkx', or 'hybrid'
    """
    
    # Get node positions for spatial clustering
    nodes_gdf, _ = ox.graph_to_gdfs(graph)
    
    if method == 'kmeans':
        # Method 1: Spatial K-Means clustering based on geographical coordinates
        coords = np.array([[node_data['y'], node_data['x']] for node_id, node_data in graph.nodes(data=True)])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
        community_labels = kmeans.fit_predict(coords)
        
        # Create community dictionary
        communities = {}
        for i, node_id in enumerate(graph.nodes()):
            communities[node_id] = community_labels[i]
            
    elif method == 'networkx':
        # Method 2: NetworkX community detection (Louvain algorithm)
        try:
            import networkx.algorithms.community as nx_comm
            communities_sets = nx_comm.greedy_modularity_communities(graph.to_undirected())
            
            communities = {}
            for i, community_set in enumerate(communities_sets[:n_communities]):
                for node in community_set:
                    communities[node] = i
        except ImportError:
            print("NetworkX community module not available, falling back to k-means")
            return perform_community_detection(graph, n_communities, 'kmeans')
    
    elif method == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split

        # Get coordinates of all nodes
        coords = np.array([[data['y'], data['x']] for _, data in graph.nodes(data=True)])
        node_ids = list(graph.nodes())

        # Step 1: create some pseudo-labels using KMeans on a small sample of nodes
        n_seed = max(50, len(coords) // 10)  # about 10% of nodes or 50 minimum
        sample_idx = np.random.choice(len(coords), size=n_seed, replace=False)
        sample_coords = coords[sample_idx]

        kmeans = KMeans(n_clusters=n_communities, random_state=42, n_init=10)
        seed_labels = kmeans.fit_predict(sample_coords)

        # Step 2: train kNN classifier on seed nodes
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(sample_coords, seed_labels)

        # Step 3: predict communities for all nodes
        predicted_labels = knn.predict(coords)

        # Build communities dict
        communities = {node_ids[i]: int(predicted_labels[i]) for i in range(len(node_ids))}
           
    elif method == 'hybrid':
        # Method 3: True Hybrid Community Detection
        # Combines topological community detection with spatial constraints
        
        # Step 1: Convert multigraph to simple graph for community detection
        simple_graph = nx.Graph(graph)
        print(f"Converting multigraph ({len(graph.edges())} edges) to simple graph ({len(simple_graph.edges())} edges)")
        
        # Step 2: Initial community detection using pure network topology
        try:
            import networkx.algorithms.community as nx_comm
            print("Running initial topological community detection...")
            
            # Use Louvain-like algorithm (greedy modularity)
            initial_communities = list(nx_comm.greedy_modularity_communities(simple_graph))
            print(f"Found {len(initial_communities)} initial topological communities")
            
            # Convert to node->community mapping
            topo_communities = {}
            for i, community_set in enumerate(initial_communities):
                for node in community_set:
                    topo_communities[node] = i
                    
        except ImportError:
            print("NetworkX community module not available, using label propagation")
            # Fallback: Label Propagation Algorithm
            topo_communities = nx.algorithms.community.label_propagation_communities(simple_graph)
            topo_communities = {node: i for i, community_set in enumerate(topo_communities) 
                              for node in community_set}
        
        # Step 3: Spatial refinement of topological communities
        print("Applying spatial refinement...")
        
        # Get node coordinates
        node_coords = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            node_coords[node_id] = (node_data['y'], node_data['x'])
        
        # Calculate spatial distances within each topological community
        refined_communities = {}
        community_id_counter = 0
        
        for topo_comm_id in set(topo_communities.values()):
            # Get nodes in this topological community
            nodes_in_topo_comm = [node for node, comm in topo_communities.items() if comm == topo_comm_id]
            
            if len(nodes_in_topo_comm) <= 2:
                # Small communities stay as is
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
                continue
            
            # Extract coordinates for this community
            comm_coords = np.array([node_coords[node] for node in nodes_in_topo_comm])
            
            # Determine number of spatial subclusters
            # Use elbow method or maximum of 3 subclusters per topological community
            max_subclusters = min(3, max(2, len(nodes_in_topo_comm) // 10))
            
            if len(nodes_in_topo_comm) > 5:
                # Apply k-means to spatially subdivide large topological communities
                kmeans = KMeans(n_clusters=max_subclusters, random_state=42, n_init=10)
                spatial_labels = kmeans.fit_predict(comm_coords)
                
                # Assign refined community IDs
                for i, node in enumerate(nodes_in_topo_comm):
                    refined_communities[node] = community_id_counter + spatial_labels[i]
                community_id_counter += max_subclusters
            else:
                # Keep small topological communities intact
                for node in nodes_in_topo_comm:
                    refined_communities[node] = community_id_counter
                community_id_counter += 1
        
        # Step 4: Post-processing to ensure we have the desired number of communities
        current_communities = len(set(refined_communities.values()))
        print(f"Refined communities: {current_communities}")
        
        if current_communities > n_communities:
            # Merge smallest communities
            community_sizes = {}
            for comm_id in set(refined_communities.values()):
                community_sizes[comm_id] = sum(1 for c in refined_communities.values() if c == comm_id)
            
            # Sort communities by size
            sorted_comms = sorted(community_sizes.items(), key=lambda x: x[1])
            communities_to_merge = sorted_comms[:current_communities - n_communities]
            
            # Merge smallest communities with their spatially nearest larger community
            for small_comm_id, _ in communities_to_merge:
                nodes_in_small = [node for node, comm in refined_communities.items() if comm == small_comm_id]
                
                if nodes_in_small:
                    # Find center of small community
                    small_coords = np.array([node_coords[node] for node in nodes_in_small])
                    small_center = np.mean(small_coords, axis=0)
                    
                    # Find nearest larger community
                    min_distance = float('inf')
                    target_comm = None
                    
                    for other_comm_id in set(refined_communities.values()):
                        if other_comm_id != small_comm_id and community_sizes[other_comm_id] > community_sizes[small_comm_id]:
                            other_nodes = [node for node, comm in refined_communities.items() if comm == other_comm_id]
                            other_coords = np.array([node_coords[node] for node in other_nodes])
                            other_center = np.mean(other_coords, axis=0)
                            
                            distance = np.linalg.norm(small_center - other_center)
                            if distance < min_distance:
                                min_distance = distance
                                target_comm = other_comm_id
                    
                    # Merge communities
                    if target_comm is not None:
                        for node in nodes_in_small:
                            refined_communities[node] = target_comm
        
        # Relabel communities to be consecutive integers starting from 0
        unique_comms = sorted(set(refined_communities.values()))
        comm_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_comms)}
        communities = {node: comm_mapping[comm_id] for node, comm_id in refined_communities.items()}
        
        print(f"Final hybrid communities: {len(set(communities.values()))}")
    
    return communities

def plot_communities(graph, communities, title="Network Communities", figsize=(15, 10)):
    """
    Plot the graph with nodes colored by community
    """
    # Create color palette
    n_communities = len(set(communities.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, n_communities))
    
    # Create node colors list
    node_colors = [colors[communities[node]] for node in graph.nodes()]
    
    # Plot the graph
    fig, ax = ox.plot_graph(
        graph,
        node_size=15,
        node_color=node_colors,
        edge_color="gray",
        bgcolor="black",
        figsize=figsize,
        show=False,
        close=False
    )
    
    ax.set_title(title, color="white", fontsize=16, pad=20)
    
    # Add legend
    legend_elements = []
    for i in range(n_communities):
        legend_elements.append(plt.scatter([], [], c=[colors[i]], s=50, label=f'Community {i+1}'))
    
    ax.legend(handles=legend_elements, loc='upper right', 
              facecolor='white', edgecolor='black')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def analyze_communities(graph, communities):
    """
    Analyze the detected communities
    """
    print("Community Analysis:")
    print("-" * 50)
    
    community_stats = {}
    for community_id in set(communities.values()):
        nodes_in_community = [node for node, comm in communities.items() if comm == community_id]
        community_stats[community_id] = {
            'size': len(nodes_in_community),
            'avg_degree': np.mean([graph.degree[node] for node in nodes_in_community])
        }
    
    for comm_id, stats in community_stats.items():
        print(f"Community {comm_id + 1}: {stats['size']} nodes, avg degree: {stats['avg_degree']:.2f}")
    
    # Calculate modularity if possible
    try:
        import networkx.algorithms.community as nx_comm
        partition = [set([node for node, comm in communities.items() if comm == comm_id]) 
                    for comm_id in set(communities.values())]
        modularity = nx_comm.modularity(graph.to_undirected(), partition)
        print(f"\nModularity Score: {modularity:.3f}")
    except:
        print("\nModularity calculation not available")

# -----------------------------
# Apply Community Detection to Venice Network
# -----------------------------

print("Performing Community Detection on Venice Network...")
print("=" * 60)

# Apply different methods
methods = ['kmeans', 'knn', 'hybrid']
n_communities = 6  # You can adjust this number

for method in methods:
    print(f"\nMethod: {method.upper()}")
    try:
        communities = perform_community_detection(G_core, n_communities=n_communities, method=method)
        
        # Plot results
        plot_communities(G_core, communities, 
                        title=f"Venice Network - {method.upper()} Communities (k={n_communities})")
        
        # Analyze communities
        analyze_communities(G_core, communities)
        
        # Save community data
        filename = f"venice_communities_{method}.txt"
        with open(filename, "w") as f:
            f.write("Node_ID\tCommunity\n")
            for node_id, community_id in communities.items():
                f.write(f"{node_id}\t{community_id}\n")
        print(f"Community data saved to {filename}")
        
    except Exception as e:
        print(f"Error with {method}: {e}")

def filter_crowded_communities(graph, communities, keep_top_n=3, min_size=None):
    """
    Filter to keep only the most crowded (largest) communities
    
    Parameters:
    - graph: NetworkX graph
    - communities: dict mapping node_id -> community_id
    - keep_top_n: number of largest communities to keep
    - min_size: minimum size threshold for communities (optional)
    
    Returns:
    - filtered_graph: NetworkX graph with only nodes from crowded communities
    - filtered_communities: dict with filtered community assignments
    - community_stats: dict with statistics about kept communities
    """
    
    print("Analyzing community sizes...")
    
    # Count nodes in each community
    community_sizes = {}
    for node, comm_id in communities.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    # Sort communities by size (largest first)
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print("Community sizes (sorted):")
    for i, (comm_id, size) in enumerate(sorted_communities):
        print(f"  Community {comm_id}: {size} nodes")
    
    # Determine which communities to keep
    if min_size is not None:
        # Keep communities above minimum size threshold
        communities_to_keep = [comm_id for comm_id, size in sorted_communities if size >= min_size]
        print(f"\nKeeping communities with >= {min_size} nodes: {len(communities_to_keep)} communities")
    else:
        # Keep top N largest communities
        communities_to_keep = [comm_id for comm_id, _ in sorted_communities[:keep_top_n]]
        print(f"\nKeeping top {keep_top_n} largest communities")
    
    if not communities_to_keep:
        print("Warning: No communities meet the criteria!")
        return graph, communities, {}
    
    # Filter nodes to keep only those in crowded communities
    nodes_to_keep = [node for node, comm_id in communities.items() if comm_id in communities_to_keep]
    print(f"Keeping {len(nodes_to_keep)} nodes out of {len(communities)} total nodes")
    
    # Create filtered graph
    filtered_graph = graph.subgraph(nodes_to_keep).copy()
    
    # Create filtered communities (re-map community IDs to be consecutive)
    filtered_communities = {}
    community_mapping = {old_id: new_id for new_id, old_id in enumerate(communities_to_keep)}
    
    for node in nodes_to_keep:
        old_comm_id = communities[node]
        filtered_communities[node] = community_mapping[old_comm_id]
    
    # Create statistics
    community_stats = {}
    for new_id, old_id in enumerate(communities_to_keep):
        nodes_in_community = [node for node, comm in filtered_communities.items() if comm == new_id]
        community_stats[new_id] = {
            'original_id': old_id,
            'size': len(nodes_in_community),
            'nodes': nodes_in_community
        }
    
    print(f"Filtered graph: {len(filtered_graph.nodes())} nodes, {len(filtered_graph.edges())} edges")
    
    return filtered_graph, filtered_communities, community_stats


def plot_crowded_communities(filtered_graph, filtered_communities, community_stats, 
                           title="Crowded Communities Only", figsize=(15, 10)):
    """
    Plot only the crowded communities with enhanced visualization
    """
    if not filtered_communities:
        print("No communities to plot!")
        return None, None
    
    print("Creating crowded communities visualization...")
    
    # Create enhanced color palette for fewer communities
    n_communities = len(set(filtered_communities.values()))
    
    # Use distinct colors for better visibility
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    if n_communities > len(colors):
        colors = plt.cm.Set3(np.linspace(0, 1, n_communities))
    
    # Create node colors and sizes based on community
    node_colors = []
    node_sizes = []
    
    for node in filtered_graph.nodes():
        comm_id = filtered_communities[node]
        node_colors.append(colors[comm_id])
        # Make nodes in larger communities slightly bigger
        community_size = community_stats[comm_id]['size']
        base_size = 20
        size_multiplier = 1 + (community_size / max([stats['size'] for stats in community_stats.values()])) * 2
        node_sizes.append(base_size * size_multiplier)
    
    # Plot the filtered graph
    fig, ax = ox.plot_graph(
        filtered_graph,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="gray",
        bgcolor="black",
        figsize=figsize,
        show=False,
        close=False
    )
    
    ax.set_title(title, color="white", fontsize=16, pad=20)
    
    # Enhanced legend with community statistics
    legend_elements = []
    for comm_id, stats in community_stats.items():
        color = colors[comm_id]
        label = f'Community {comm_id + 1}: {stats["size"]} nodes'
        legend_elements.append(plt.scatter([], [], c=[color], s=80, label=label))
    
    ax.legend(handles=legend_elements, loc='upper right', 
              facecolor='white', edgecolor='black', fontsize=10)
    
    # Add text box with summary statistics
    total_nodes = sum(stats['size'] for stats in community_stats.values())
    summary_text = f"Total nodes in crowded communities: {total_nodes}\n"
    summary_text += f"Number of communities shown: {n_communities}\n"
    summary_text += f"Largest community: {max(stats['size'] for stats in community_stats.values())} nodes"
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9, color='black')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def analyze_crowded_communities(filtered_graph, filtered_communities, community_stats, original_graph):
    """
    Provide detailed analysis of the crowded communities
    """
    print("\n" + "="*60)
    print("CROWDED COMMUNITIES ANALYSIS")
    print("="*60)
    
    total_original_nodes = len(original_graph.nodes())
    total_filtered_nodes = len(filtered_graph.nodes())
    coverage_percentage = (total_filtered_nodes / total_original_nodes) * 100
    
    print(f"Coverage: {total_filtered_nodes}/{total_original_nodes} nodes ({coverage_percentage:.1f}%)")
    print(f"Removed {total_original_nodes - total_filtered_nodes} nodes from smaller communities")
    
    print("\nDetailed Community Statistics:")
    for comm_id, stats in community_stats.items():
        nodes_in_comm = [node for node, comm in filtered_communities.items() if comm == comm_id]
        
        # Calculate network properties for this community
        subgraph = filtered_graph.subgraph(nodes_in_comm)
        avg_degree = np.mean([filtered_graph.degree[node] for node in nodes_in_comm]) if nodes_in_comm else 0
        
        # Calculate density
        n_nodes = len(nodes_in_comm)
        n_edges = len(subgraph.edges())
        max_possible_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        print(f"\nCommunity {comm_id + 1} (Original ID: {stats['original_id']}):")
        print(f"  Size: {stats['size']} nodes")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Internal edges: {n_edges}")
        print(f"  Density: {density:.3f}")
        print(f"  Percentage of total: {(stats['size']/total_filtered_nodes)*100:.1f}%")
    
    return {
        'coverage_percentage': coverage_percentage,
        'total_filtered_nodes': total_filtered_nodes,
        'total_original_nodes': total_original_nodes
    }


# Usage example - Add this to the main execution section
print("\n" + "="*60)
print("FILTERING TO CROWDED COMMUNITIES ONLY")
print("="*60)

# Apply this to the best performing method (you can change this)
best_method = 'knn'  # or 'hybrid', 'kmeans', etc.

try:
    print(f"Using communities from {best_method.upper()} method...")
    
    # Get communities using the best method
    best_communities = perform_community_detection(G_core, n_communities=6, method=best_method)
    
    if best_communities:
        # Filter to keep only crowded communities
        print("\nFiltering to keep top 3 largest communities:")
        filtered_graph, filtered_communities, community_stats = filter_crowded_communities(
            G_core, best_communities, keep_top_n=3
        )
        
        # Visualize crowded communities
        plot_crowded_communities(
            filtered_graph, filtered_communities, community_stats,
            title=f"Venice Network - Top 3 Crowded Communities ({best_method.upper()})"
        )
        
        # Analyze the results
        analysis_results = analyze_crowded_communities(
            filtered_graph, filtered_communities, community_stats, G_core
        )
        
        # Also try with minimum size threshold
        print(f"\n{'-'*40}")
        print("Alternative: Filtering by minimum size (50+ nodes):")
        filtered_graph_alt, filtered_communities_alt, community_stats_alt = filter_crowded_communities(
            G_core, best_communities, keep_top_n=None, min_size=50
        )
        
        if filtered_communities_alt:
            plot_crowded_communities(
                filtered_graph_alt, filtered_communities_alt, community_stats_alt,
                title=f"Venice Network - Large Communities (50+ nodes, {best_method.upper()})"
            )
        
        # Save filtered results
        print("\nSaving filtered community data...")
        with open(f"venice_crowded_communities_{best_method}.txt", "w") as f:
            f.write("Node_ID\tCommunity\tCommunity_Size\n")
            for node_id, community_id in filtered_communities.items():
                community_size = community_stats[community_id]['size']
                f.write(f"{node_id}\t{community_id}\t{community_size}\n")
        
        print(f"Crowded communities data saved to venice_crowded_communities_{best_method}.txt")
    
except Exception as e:
    print(f"Error in crowded communities analysis: {e}")

def plot_crowded_vs_rest(graph, communities, keep_top_n=3, min_size=None, 
                        title="Crowded Communities (Red) vs Rest (White)", figsize=(15, 10)):
    """
    Plot the entire graph with crowded communities in RED and rest in WHITE
    
    Parameters:
    - graph: NetworkX graph (complete graph)
    - communities: dict mapping node_id -> community_id
    - keep_top_n: number of largest communities to highlight in red
    - min_size: minimum size threshold for communities (alternative to keep_top_n)
    - title: plot title
    - figsize: figure size
    """
    
    print("Creating crowded vs rest visualization...")
    
    # Count nodes in each community
    community_sizes = {}
    for node, comm_id in communities.items():
        community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
    
    # Sort communities by size (largest first)
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Determine which communities are "crowded"
    if min_size is not None:
        crowded_community_ids = [comm_id for comm_id, size in sorted_communities if size >= min_size]
        print(f"Crowded communities (>= {min_size} nodes): {len(crowded_community_ids)} communities")
    else:
        crowded_community_ids = [comm_id for comm_id, _ in sorted_communities[:keep_top_n]]
        print(f"Crowded communities (top {keep_top_n}): {crowded_community_ids}")
    
    # Print community sizes for reference
    print("Community sizes:")
    for comm_id, size in sorted_communities:
        status = "CROWDED" if comm_id in crowded_community_ids else "small"
        print(f"  Community {comm_id}: {size} nodes ({status})")
    
    # Create node colors: RED for crowded communities, WHITE for rest
    node_colors = []
    crowded_nodes = 0
    rest_nodes = 0
    
    for node in graph.nodes():
        if node in communities:
            comm_id = communities[node]
            if comm_id in crowded_community_ids:
                node_colors.append('red')
                crowded_nodes += 1
            else:
                node_colors.append('white')
                rest_nodes += 1
        else:
            # Node not in any community (shouldn't happen, but safety check)
            node_colors.append('gray')
            rest_nodes += 1
    
    print(f"Visualization: {crowded_nodes} RED nodes, {rest_nodes} WHITE nodes")
    
    # Plot the complete graph
    fig, ax = ox.plot_graph(
        graph,
        node_size=12,  # Slightly larger nodes for better visibility
        node_color=node_colors,
        edge_color="lightgray",  # Lighter edges to make nodes stand out
        bgcolor="black",
        figsize=figsize,
        show=False,
        close=False
    )
    
    ax.set_title(title, color="white", fontsize=16, pad=20)
    
    # Create legend
    red_patch = plt.scatter([], [], c='red', s=80, label=f'Crowded Communities ({crowded_nodes} nodes)')
    white_patch = plt.scatter([], [], c='white', s=80, edgecolors='gray', 
                             label=f'Other Communities ({rest_nodes} nodes)')
    
    ax.legend(handles=[red_patch, white_patch], loc='upper right', 
              facecolor='black', edgecolor='white', fontsize=12)
    
    # Add statistics text box
    total_nodes = len(graph.nodes())
    coverage_percent = (crowded_nodes / total_nodes) * 100
    
    stats_text = f"Total nodes: {total_nodes}\n"
    stats_text += f"Crowded nodes: {crowded_nodes} ({coverage_percent:.1f}%)\n"
    stats_text += f"Other nodes: {rest_nodes} ({100-coverage_percent:.1f}%)\n"
    stats_text += f"Crowded communities: {len(crowded_community_ids)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()
    
    # Return statistics for further analysis
    return {
        'crowded_nodes': crowded_nodes,
        'rest_nodes': rest_nodes,
        'crowded_community_ids': crowded_community_ids,
        'coverage_percent': coverage_percent,
        'total_communities': len(set(communities.values())),
        'crowded_communities': len(crowded_community_ids)
    }


def compare_crowded_highlighting(graph, communities, methods_to_compare=['top3', 'top5', 'min50', 'min100']):
    """
    Show multiple versions of crowded vs rest visualization for comparison
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    configs = {
        'top3': {'keep_top_n': 3, 'min_size': None, 'title': 'Top 3 Largest Communities'},
        'top5': {'keep_top_n': 5, 'min_size': None, 'title': 'Top 5 Largest Communities'}, 
        'min50': {'keep_top_n': None, 'min_size': 50, 'title': 'Communities with 50+ Nodes'},
        'min100': {'keep_top_n': None, 'min_size': 100, 'title': 'Communities with 100+ Nodes'}
    }
    
    for i, method in enumerate(methods_to_compare):
        if i >= 4:  # Only 4 subplots available
            break
            
        config = configs[method]
        ax = axes[i]
        
        # Count community sizes
        community_sizes = {}
        for node, comm_id in communities.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Determine crowded communities
        if config['min_size'] is not None:
            crowded_ids = [comm_id for comm_id, size in sorted_communities if size >= config['min_size']]
        else:
            crowded_ids = [comm_id for comm_id, _ in sorted_communities[:config['keep_top_n']]]
        
        # Create colors
        node_colors = []
        for node in graph.nodes():
            if node in communities and communities[node] in crowded_ids:
                node_colors.append('red')
            else:
                node_colors.append('white')
        
        # Get node positions
        pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
        
        # Draw on subplot
        nx.draw_networkx(graph, pos, ax=ax, node_color=node_colors,
                        node_size=8, edge_color='lightgray', with_labels=False)
        
        crowded_count = sum(1 for color in node_colors if color == 'red')
        ax.set_title(f'{config["title"]}\n{crowded_count} red nodes, {len(crowded_ids)} communities', 
                    fontsize=11, color='white')
        ax.set_facecolor('black')
    
    plt.suptitle('Venice Network: Different Crowded Community Definitions', 
                 fontsize=16, color='white', y=0.95)
    plt.tight_layout()
    plt.show()


# Usage - Add this to the main execution section
print("\n" + "="*60)
print("RED vs WHITE CROWDED COMMUNITIES VISUALIZATION")
print("="*60)

try:
    # Use the best method results
    best_method = 'knn'
    best_communities = perform_community_detection(G_core, n_communities=6, method=best_method)
    
    if best_communities:
        print(f"Using {best_method.upper()} method results...")
        
        # Method 1: Show top 3 largest communities in RED
        print("\n1. Highlighting TOP 3 largest communities in RED:")
        stats1 = plot_crowded_vs_rest(
            G_core, best_communities, keep_top_n=3,
            title=f"Venice Network - Top 3 Crowded Communities in RED ({best_method.upper()})"
        )
        
        # Method 2: Show communities with 50+ nodes in RED
        print("\n2. Highlighting communities with 50+ nodes in RED:")
        stats2 = plot_crowded_vs_rest(
            G_core, best_communities, min_size=50,
            title=f"Venice Network - Large Communities (50+ nodes) in RED ({best_method.upper()})"
        )
        
        # Method 3: Comparison view
        print("\n3. Comparison of different crowded community definitions:")
        compare_crowded_highlighting(G_core, best_communities)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Top 3 approach: {stats1['crowded_nodes']} nodes ({stats1['coverage_percent']:.1f}%) in red")
        print(f"50+ nodes approach: {stats2['crowded_nodes']} nodes ({stats2['coverage_percent']:.1f}%) in red")
        
    else:
        print("No communities found!")
        
except Exception as e:
    print(f"Error in red vs white visualization: {e}")

print("\n" + "="*60)
print("RED vs WHITE VISUALIZATION COMPLETED!")
print("="*60)

'''