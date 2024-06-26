#!/usr/bin/env python3

from custom_items import *
from viewer import *
import numpy as np
from gau_io import *

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'viewer'))


if __name__ == '__main__':
    from pypcd import pypcd
    pcd_fn = "/home/liu/bag/test.pcd"
    pcd = pypcd.PointCloud.from_path(pcd_fn)
    points = np.array([pcd.pc_data["x"], pcd.pc_data["y"], pcd.pc_data["z"],
                      pcd.pc_data['intensity']/100.], dtype=np.float32).T
    app = QApplication([])
    cloud_item = CloudItem()
    grid_item = GridItem()
    items = {"cloud": cloud_item, "grid": grid_item}
    viewer = Viewer(items)
    viewer.items["cloud"].setData(pos=points)
    viewer.show()
    app.exec_()
