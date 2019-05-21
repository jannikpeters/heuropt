from model import TTSP
import numpy as np
import pandas as pd
import numpy.random
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from thief_heuristics import read_init_solution_for
from scipy.spatial import KDTree
import matplotlib


def node_locations(ttsp, name, ax):
    x = ttsp.node_coord[:, 0]
    y = ttsp.node_coord[:, 1]
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap.T, extent=extent, origin='lower')
    ax.set_title(name)


def hist1D(x, name, ax):
    #plt.hist(x)
    # only uncomment, will be slow after
    sns.distplot(x, rug=True, kde=False, ax=ax)
    ax.set_title(name)


def wp_dist(ttsp, name, z, ax):
    # plot distributions of item weight/price
    item_coords = np.zeros((len(ttsp.item_weight), 2))
    get_node_coords = lambda x: (ttsp.node_coord[x, 0], ttsp.node_coord[x, 1])
    x, y = np.vectorize(get_node_coords)(ttsp.item_node)
    rgba_colors = np.zeros((len(ttsp.item_weight), 4))
    # for red the first column needs to be one
    rgba_colors[:, 2] = 1.0
    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = z / z.max()
    ax.scatter(x, y, c=rgba_colors, s=0.5)
    ax.set_title(name)


def item_dist(ttsp, name, ax):
    # plot distributions of item weight/price
    item_coords = np.zeros((len(ttsp.item_weight), 2))
    get_node_coords = lambda x: (ttsp.node_coord[x, 0], ttsp.node_coord[x, 1])
    x, y = np.vectorize(get_node_coords)(ttsp.item_node)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.imshow(heatmap.T, extent=extent, origin='lower')
    ax.set_title(name)


def show_colorbar(cm, ax):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cm))



def add_item(city_i: int, bit_string: np.ndarray, ttsp: TTSP):
    indexes_items_in_city = np.where(ttsp.item_node == city_i)[0]
    if len(indexes_items_in_city) == 0:
        return 0, 0
    is_taken = bit_string[indexes_items_in_city]
    weights = ttsp.item_weight[indexes_items_in_city]
    weight = np.multiply(is_taken, weights).sum(dtype=np.int)

    profits = ttsp.item_profit[indexes_items_in_city]
    profit = np.multiply(is_taken, profits).sum(dtype=np.int)
    return weight, profit


def pickup_points(ttsp, tour, packing_bitstring, file):
    _, axes_row = plt.subplots(1,2, figsize=(12, 5))
    cumweight = np.zeros(tour.shape)
    cumdist = np.zeros(tour.shape)
    cumprofit = np.zeros(tour.shape)

    R = ttsp.renting_ratio
    n = len(tour)
    current_dist = 0
    current_weight = 0
    current_profit = 0
    for i in range(n):
        city_i = tour[i % n]
        city_ip1 = tour[(i + 1) % n]
        w, p = add_item(city_i, packing_bitstring, ttsp)
        current_weight += w
        current_profit += p
        cumweight[i] = current_weight
        cumprofit[i] = current_profit
        if i != 0:
            current_dist += ttsp.dist(city_i, city_ip1)
        cumdist[i] = current_dist
    axes_row[0].plot(cumdist, cumweight, label='weights')
    axes_row[0].set_xlabel('distance (s)')
    axes_row[0].set_ylabel('weight', color='blue')
    axes_row[0].tick_params(axis='y', color='blue')

    ax2 = axes_row[0].twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(cumdist, cumprofit, label='weights', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('profit', color='red')
    axes_row[0].set_title('[' + file + '] Profits vs weight over distance' )


    axes_row[1].plot(cumdist[10:], cumprofit[10:] / cumweight[10:])
    axes_row[1].set_xlabel('distance to last node')
    axes_row[1].set_ylabel('profit/weight')
    axes_row[1].set_title('Profit/Weight (of current kp) over distance')



def show_tour(ttsp, tour, file, ax):
    get_node_coords = lambda x: (ttsp.node_coord[x, 0], ttsp.node_coord[x, 1])
    x, y = np.vectorize(get_node_coords)(tour)

    fig = plt.figure()
    from matplotlib import cm
    T = cm.get_cmap('hsv')(np.linspace(0, 1, np.size(x)))

    n = len(x)
    # Segement plot and colour depending on T
    s = int(n / 100)  # Segments length
    for i in range(0, n - s, s):
        ax.plot(x[i:i + s + 1], y[i:i + s + 1], color=T[i])
    ax.plot(x[s * 100:], y[s * 100:], color=T[i])
    # ax.plot([x[-1],x[0]], [y[-1],y[0]], color=T[i])
    ax.set_title(file + '[r->g->b->r]')
    ax.scatter([x[0]], [y[0]], color=[0, 0, 0], s=100)


def read_from_file(file):
    file = str(file)
    solution_file = 'solutions/' + file.split('/')[1].split('.')[0] + '.txt'
    ttsp = TTSP(file)

    # plot distributions of item weight/price
    item_coords = np.zeros((ttsp.item_weight, 2))
    get_node_coords = lambda x: (ttsp.node_coord[x, 0], ttsp.node_coord[x, 1])
    x, y = np.vectorize(get_node_coords)(ttsp.item_node)


def far_away_neigbours(ttsp,tour,tour_coords,file, ax):
    node_pos_in_tour = np.zeros(tour.shape)
    for i in range(len(tour)):
        node = tour[i]
        node_pos_in_tour[node] = i

    tree = KDTree(ttsp.node_coord)

    for node, node_pos_in in enumerate(node_pos_in_tour):
        node_coord = ttsp.node_coord[node, :]
        distances, idx = tree.query(node_coord, k=5)
        for i in idx:
            i_coord = tree.data[i]
            i_pos_in_tour = node_pos_in_tour[i]
            if abs(node_pos_in - i_pos_in_tour) > 100:
                ax.plot([i_coord[0], node_coord[0]], [i_coord[1], node_coord[1]], color='red')
    ax.plot(tour_coords[0], tour_coords[1], color='gray', alpha=0.5)
    plt.show()

if __name__ == '__main__':

    rootdir = Path('gecc')
    files = [f for f in rootdir.glob('**/*') if f.is_file()]



    for file in [str(f) for f in files]:
        f = file.split('/')[-1].split('.')[-2]
        ttsp, kp, tour = read_init_solution_for('solutions',f)

        get_node_coords = lambda x: (ttsp.node_coord[x, 0], ttsp.node_coord[x, 1])
        tour_coords = np.vectorize(get_node_coords)(tour)

        fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(25, 25))

        # data set properties
        node_locations(ttsp, 'Node locations ' + file,axs[0,0])
        item_dist(ttsp, 'Item locations ' + file,axs[0,1])
        axs[0,2].text(0, 0.5, file, size=30, bbox=dict(facecolor='blue', alpha=0.2))

        hist1D(ttsp.item_weight, 'Histogram of: ... Weight ' ,axs[1,0])
        hist1D(ttsp.item_profit, 'Profit' ,axs[1,1])
        hist1D(ttsp.item_profit / ttsp.item_weight, 'Weight/Profit' + file,axs[1,2])

        wp_dist(ttsp, 'Alpha values for each ... Weights ', z=ttsp.item_weight, ax=axs[2,0])
        wp_dist(ttsp, 'Profits', z=ttsp.item_profit,ax=axs[2,1])
        wp_dist(ttsp, 'Profit/Weight', z=ttsp.item_profit / ttsp.item_weight,ax=axs[2,2])
        plt.title('Data Properties [' + file + ']')
        plt.show()#



        # tour evaluation
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(30, 10),gridspec_kw={'width_ratios': [3, 3,1]})
        show_tour(ttsp, tour, 'Tour '+file,axs[1])
        show_colorbar('hsv',axs[2])
        far_away_neigbours(ttsp, tour, tour_coords, file,axs[0])
        plt.show()


        # kp eval
        pickup_points(ttsp, tour, kp, file)




