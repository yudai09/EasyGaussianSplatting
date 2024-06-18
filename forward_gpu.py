import torch
import gsplatcu as gsc
import numpy as np
import matplotlib.pyplot as plt
from gsplat.gau_io import *
from gsplat.gausplat import *
from utils import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the gs path")
    args = parser.parse_args()

    if args.gs:
        print("Try to load %s ..." % args.gs)
        gs = load_gs(args.gs)
    else:
        print("not gs file.")
        gs = get_example_gs()

    # Camera info
    # tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    tcw = np.array([1.03796196, 0.42017467, 3.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(1280)
    height = int(1280)

    hfov = np.pi * 0.8
    # hfov = np.pi * 0.3

    tanx = np.tan(hfov / 2)
    tany = tanx
    # height / 2 = fx * theta
    # fx = height / (theta * 2)
    focal = width / hfov

    fx = focal
    fy = focal
    cx = width / 2
    cy = height / 2

    gs = remove_invalid_gs(gs)
    gs = remove_outof_fov(gs, hfov, tcw, Rcw)

    pws = torch.from_numpy(gs['pw']).type(torch.float32).to('cuda')
    rots = torch.from_numpy(gs['rot']).type(torch.float32).to('cuda')
    scales = torch.from_numpy(gs['scale']).type(torch.float32).to('cuda')
    alphas = torch.from_numpy(gs['alpha']).type(torch.float32).to('cuda')
    shs = torch.from_numpy(gs['sh']).type(torch.float32).to('cuda')
    Rcw = torch.from_numpy(Rcw).type(torch.float32).to('cuda')
    tcw = torch.from_numpy(tcw).type(torch.float32).to('cuda')
    twc = torch.linalg.inv(Rcw)@(-tcw)

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs, depths = gsc.project(pws, Rcw, tcw, fx, fy, cx, cy, False)

    # step2. Calcuate the 3d Gaussian.
    cov3ds = gsc.computeCov3D(rots, scales, depths, False)[0]

    # step3. Calcuate the 2d Gaussian.
    # cov2ds = gsc.computeCov2D(cov3ds, pcs, Rcw, depths, fx, fy, height, width, False)[0]
    cov2ds = gsc.computeCov2D(cov3ds, pcs, Rcw, depths, fx, fy, tanx, tany, False)[0]

    # step4. get color info
    colors = gsc.sh2Color(shs, pws, twc, False)[0]

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = gsc.inverseCov2D(cov2ds, depths, False)
    image = gsc.splat(height, width, us, cinv2ds, alphas, depths, colors, areas)[0]
    image = image.to('cpu').numpy()

    from PIL import Image
    image = image.transpose(1, 2, 0)
    print(image.shape)
    pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    pil_img.save('gs_image_gpu.png')

    plt.imshow(image)
    plt.show()
