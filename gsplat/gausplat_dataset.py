from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
from gsplat.read_write_model import *
# from read_write_model import *
from PIL import Image
import torch
import torchvision
from plyfile import PlyData
import cv2
from equidistance import convert_to_equidistant_image


class Camera:
    def __init__(self, id, width, height, fx, fy, cx, cy, Rcw, tcw):
        self.id = id
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.Rcw = Rcw
        self.tcw = tcw
        self.twc = -torch.linalg.inv(Rcw) @ tcw


class GSplatDataset(Dataset):
    def __init__(self, path, device='cuda') -> None:
        super().__init__()
        self.path = path
        self.device = device
        camera_params, image_params = read_model(Path(path, "sparse/0"), ext='.bin')
        self.camera_params = camera_params
        self.image_params = list(image_params.values())
        try:
            self.gs = np.load(Path(path, "sparse/0/points3D.npy"))
        except:
            self.gs = read_points_bin_as_gau(Path(path, "sparse/0/points3D.bin"))
            np.save(Path(path, "sparse/0/points3D.npy"), self.gs)

    def __getitem__(self, index: int):
        return self.get_camera_image(index)

    def get_camera_image(self, index: int):
        image_param = self.image_params[index]
        i = image_param.camera_id
        camera_param = self.camera_params[i]

        # image = torchvision.io.read_image(
        #     str(Path(self.path, "images", image_param.name))).to(self.device).to(torch.float32) / 255.
        image = cv2.imread(str(Path(self.path, "images", image_param.name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert to equidistant fisheye image
        image, camera_param = convert_to_equidistant_image(image, camera_param)
        image = torch.tensor(image).permute(2, 0, 1).to(torch.float32).to(self.device) / 255.

        _, height, width = image.shape
        w_scale = width/camera_param.width
        h_scale = height/camera_param.height
        fx = camera_param.params[0] * w_scale
        fy = camera_param.params[1] * h_scale
        cx = camera_param.params[2] * w_scale
        cy = camera_param.params[3] * h_scale
        Rcw = torch.from_numpy(image_param.qvec2rotmat()).to(self.device).to(torch.float32)
        tcw = torch.from_numpy(image_param.tvec).to(self.device).to(torch.float32)
        camera = Camera(image_param.id, width, height, fx, fy, cx, cy, Rcw, tcw)

        return camera, image

    def __len__(self) -> int:
        return len(self.image_params)


if __name__ == "__main__":
    path = '/home/liu/bag/gaussian-splatting/tandt/train'
    gs_dataset = GSplatDataset(path)
    gs_dataset[0]
