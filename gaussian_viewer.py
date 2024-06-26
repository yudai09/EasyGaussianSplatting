#!/usr/bin/env python3

import numpy as np
from gsplat.gau_io import *

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'viewer'))

from viewer import *
from custom_items import *


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the input gs path")
    args = parser.parse_args()
    cam_2_world = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    if args.gs:
        print("Try to load %s ..." % args.gs)
        gs = load_gs(args.gs)
        rotate_gaussian(cam_2_world, gs)
    else:
        print("not gs file.")
        gs = get_example_gs()

    gs_data = gs.view(np.float32).reshape(gs.shape[0], -1)

    app = QApplication([])
    gs_item = GaussianItem()
    grid_item = GridItem()

    items = {"gs": gs_item, "grid": grid_item}

    viewer = Viewer(items)
    viewer.items["gs"].setData(gs_data=gs_data)
    viewer.show()
    app.exec_()
