import numpy as np
from collections import OrderedDict
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d

##################################################################################
# The following group of functions assists in setting up a nice environment for 
# making figures in the examples that we provide. - Kevin Quinn
#
#
###################################################################################




########################################################################################
# The following functions are used for creating custom color maps (which I use for my
# geographical maps). Code is taken from https://gist.github.com/aviputri
# and new color maps may be found online.
#
########################################################################################

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp






class graph_util():
    def __init__(self):
        pass
        
    def linestyles(self):
        # simply store and return this dictionary 
        # (allows for further functionality in future development)
        linestyles_dict = OrderedDict(
            [('solid',               (0, ())),
            ('loosely dotted',      (0, (1, 10))),
            ('dotted',              (0, (1, 5))),
            ('densely dotted',      (0, (1, 1))),

            ('loosely dashed',      (0, (5, 10))),
            ('dashed',              (0, (5, 5))),
            ('densely dashed',      (0, (5, 1))),

            ('loosely dashdotted',  (0, (3, 10, 1, 10))),
            ('dashdotted',          (0, (3, 5, 1, 5))),
            ('densely dashdotted',  (0, (3, 1, 1, 1))),

            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
        
        return linestyles_dict
    
    def custom_map(self, hex_list):
        return get_continuous_cmap(hex_list)
    
    
        