import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import matplotlib

rad = .2

def offset(d, pos, dist = rad/2, loop_shift = .2):
    for (u,v),obj in d.items():
        if u!=v:
            par = dist*(pos[v] - pos[u])
            dx,dy = par[1],-par[0]
            x,y = obj.get_position()
            obj.set_position((x+dx,y+dy))
        else:
            x,y = obj.get_position()
            obj.set_position((x,y+loop_shift))



def draw_graph(edge_index, edge_values=None):
        edge_list = edge_index.numpy().T.tolist()
        conn_style = f'arc3,rad={rad}'

        G = nx.MultiDiGraph()
        G.add_edges_from(edge_list)


        pos = nx.spring_layout(G) 


        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
        nx.draw_networkx_labels(G, pos=pos, font_color='red')
        nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), edge_color='black',
                        connectionstyle=conn_style)
        
        if edge_values is not None:
        # Draw edges with colors based on the starting node
                edge_value_list = edge_values.numpy().tolist() 
                edge_labels = {tuple(edge): f'{value:.2f}' for edge, value in zip(edge_list, edge_value_list)}
                d = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
                offset(d,pos)
        plt.gca().set_aspect('equal')

        plt.show()


def plot_embeddings_overtime(graph_data, embeddings, plotting_epochs):
    fig, axes = plt.subplots(1, len(embeddings), figsize=(20, 5))
    for i, emb in enumerate(embeddings):
        emb = emb.detach().numpy()
        ax = axes[i]

        graph = to_networkx(graph_data, to_undirected=False)
        pos = {j: e for j,e in enumerate(emb)}
        colormap = matplotlib.colormaps['viridis']

        nx.draw(
            graph,
            pos,
            node_color=graph_data.y,
            node_size=100,
            ax=ax,
            cmap =colormap,
            with_labels=False
        )
        ax.set_title(f'Embeddings at epoch {int(plotting_epochs[i])}')
        ax.set_xlim(emb[:, 0].min() - 0.1, emb[:, 0].max() + 0.1)
        ax.set_ylim(emb[:, 1].min() - 0.1, emb[:, 1].max() + 0.1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    for ax in axes:
        ax.axis('on')
    plt.tight_layout()
    plt.show()


def plot_graph_and_subgraph(G, G_sub, fixed_layout, title=None):
    subgraph_layout = {node: fixed_layout[node] for node in G_sub.nodes()}

    all_x, all_y = zip(*fixed_layout.values())
    x_limits = (min(all_x)-.1, max(all_x)+.1)
    y_limits = (min(all_y)-.1, max(all_y)+.1)

    # Draw the full graph and the subgraph side by side with fixed axis limits
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the full graph on the left
    nx.draw(G, pos=fixed_layout, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, ax=ax1)
    ax1.set_title(r'Original graph using $A$')
    ax1.set_xlim(x_limits)
    ax1.set_ylim(y_limits)

    # Plot the subgraph on the right using the same axis limits
    nx.draw(G_sub, pos=subgraph_layout, with_labels=True, node_color="lightgreen", edge_color="gray", node_size=500, ax=ax2)
    ax2.set_title(title)
    ax2.set_xlim(x_limits)
    ax2.set_ylim(y_limits)

    plt.tight_layout()
    plt.show()