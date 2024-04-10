import itertools
import json
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

from Kohonen_SOM.som import SOM
from Kohonen_SOM.som import plot_data_on_map


TYPE_COLORS = {
    'bug': 'g',
    'dragon': 'b',
    'electric': 'y',
    'fairy': 'tab:pink',
    'fighting': 'maroon',
    'fire': 'tab:red',
    'flying': 'salmon',
    'ghost': 'darkorchid',
    'grass': 'g',
    'ground': 'tab:brown',
    'ice': 'tab:cyan',
    'normal': 'k',
    'poison': 'tab:purple',
    'psychic': 'plum',
    'rock': 'tab:gray',
    'steel': 'lightsteelblue',
    'water': 'b',
}

# Type: (Symbol, Colors)
MARKER_BY_TYPE = {
    'bug': ('Bug', 'g'),
    'dragon': ('Drg', 'b'),
    'electric': ('Elec', 'y'),
    'fairy': ('Far', 'tab:pink'),
    'fighting': ('Fgt', 'maroon'),
    'fire': ("Fire", 'tab:red'),
    'flying': ("Fly", "salmon"),
    'ghost': ("Gho", "darkorchid"),
    'grass': ("Grs", "g"),
    'ground': ("Gnd", "tab:brown"),
    'ice': ("Ice", "tab:cyan"),
    'normal': ("Nor", "k"),
    'poison': ("Poi", "tab:purple"),
    'psychic': ("Psy", "plum"),
    'rock': ("Rock", "tab:gray"),
    'steel': ("Stl", "lightsteelblue"),
    'water': ("Wtr", "b"),
}


def marker_plot(som_locations, unique_types, indicies_by_type):
    fig, ax = plt.subplots()
    fig.suptitle("Title goes here!")
    for type_, (symbol, color) in MARKER_BY_TYPE.items():
        blah = som_locations[indicies_by_type[type_]]
        pts, counts = np.unique(blah, axis=0, return_counts=True)
        for count, pt in zip(counts, pts):
            t = Affine2D().scale(count)
            m = MarkerStyle(TextPath((0, 0), symbol[0]), transform=t)
            ax.plot(pt[0], pt[1], marker=m, color=color)
    plt.show()

def get_stats(m):
    return {_s["stat"]["name"]: _s["base_stat"] for _s in m["stats"]}

def main():
    pokedex = json.load(open("gen1.json", "r"))
    STAT_NAMES = list(get_stats(pokedex[0]).keys())
    stats_dict = [get_stats(_p) for _p in pokedex]
    stats = np.array([[_s[_n] for _n in STAT_NAMES] for _s in stats_dict])
    assert len(stats_dict) == len(stats)
    types_ = [[_t["type"]["name"] for _t in _p["types"]] for _p in pokedex]
    assert len(types_) == len(stats)
    TYPES = set(itertools.chain.from_iterable(types_))
    som = SOM()
    assert stats.shape[-1] == 6
    som.create(9, 9, stats.shape[-1])
    som.fit(stats, random_sampling=1, n_iter=50)
    data_locations, vect_distances = som.find_maching_nodes(stats)
#    map_vectors = som.get_map_vectors().copy()
#    map_vectors -= np.min(map_vectors)
#    map_vectors /= np.max(map_vectors)
#    plt.imshow(map_vectors)
#    umatrix = som.get_umatrix()
#    plt.imshow(umatrix, cmap="gray")
#    plt.show()
    indicies_by_type = {_ty: [_i for _i, _p in enumerate(pokedex) if _ty in [_t["type"]["name"] for _t in _p["types"]]] for _ty in TYPES}
    #plt.hist2d(x, y, bins=8, density=True, cmap="Grays")
    #plt.title(demo_type.capitalize())
    #plt.show()
#    marker_plot(data_locations, TYPES, indicies_by_type)
    plot_data_on_map(
        umatrix=som.get_umatrix(),
        data_locations=data_locations,
        data_colors=[TYPE_COLORS[t[0]] for t in types_],
        data_labels=types_,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='SOMoMons',
                    description='Self-Organizing Maps of \'Mons',
                    epilog='E.g., Pokémon')
    # TODO argument for (re-)downloading data
    main()
