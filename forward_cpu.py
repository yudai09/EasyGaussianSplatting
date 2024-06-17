from gsplat.gausplat import *
from gsplat.gau_io import *


def remove_invalid_gs(gs):
    pws = gs['pw']
    x = pws[:, 0]
    y = pws[:, 1]
    z = pws[:, 2]
    valid_inds = np.logical_not(np.isnan(x + y + z))
    keys = ['pw', 'scale', 'sh', 'rot', 'alpha']
    new_gs = {}
    for key in keys:
        new_gs[key] = gs[key][valid_inds]
    return new_gs


def remove_outof_fov(gs, fov, tcw, Rcw):
    pws = gs['pw']
    x = pws[:, 0]
    y = pws[:, 1]
    z = pws[:, 2]

    r = np.sqrt(x * x + y * y) + 1e-3
    # insident angle
    theta = np.arctan2(r, z)

    # valid_inds = np.logical_and(theta < fov, theta > 0)

    # temporary solution
    # FIXME: 
    pc = (Rcw @ pws.T).T + tcw
    valid_inds = pc[:, 2] > 0

    keys = ['pw', 'scale', 'sh', 'rot', 'alpha']
    new_gs = {}
    for key in keys:
        new_gs[key] = gs[key][valid_inds]
    return new_gs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", help="the input 3d gaussians path")
    args = parser.parse_args()

    if args.gs:
        ply_fn = args.gs
        print("Try to load %s ..." % ply_fn)
        gs = load_gs(ply_fn)
    else:
        print("not gaussians file.")
        gs = get_example_gs()

    # Camera info
    tcw = np.array([1.03796196, 0.42017467, 4.67804612])
    # tcw = np.array([0.53796196, 0.42017467, 2.67804612])
    Rcw = np.array([[0.89699204,  0.06525223,  0.43720409],
                    [-0.04508268,  0.99739184, -0.05636552],
                    [-0.43974177,  0.03084909,  0.89759429]]).T

    width = int(1280)
    height = int(1280)

    # hfov = np.pi * 0.8
    hfov = np.pi * 0.3

    tanx = np.tan(hfov / 2)
    tany = tanx
    # height / 2 = fx * theta
    # fx = height / (theta * 2)
    focal = width / hfov

    fx = focal
    fy = focal
    cx = width / 2
    cy = height / 2

    twc = np.linalg.inv(Rcw) @ (-tcw)

    fig, ax = plt.subplots()
    array = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    im = ax.imshow(array)

    gs = remove_invalid_gs(gs)
    gs = remove_outof_fov(gs, hfov, tcw, Rcw)

    pws = gs['pw']

    # step1. Transform pw to camera frame,
    # and project it to iamge.
    us, pcs = project(pws, Rcw, tcw, fx, fy, cx, cy)
    # print(us)
    # import matplotlib.pyplot as plt
    # plt.plot(us[:, 0], us[:, 1])

    # depths = pcs[:, 2]
    depths = np.linalg.norm(pcs, axis=1)

    # step2. Calcuate the 3d Gaussian.
    cov3ds = compute_cov_3d(gs['scale'], gs['rot'])

    # step3. Project the 3D Gaussian to 2d image as a 2d Gaussian.
    cov2ds = compute_cov_2d(pcs, fx, fy, width, height, cov3ds, Rcw)

    # step4. get color info
    colors = sh2color(gs['sh'], pws, twc)

    # step5. Blend the 2d Gaussian to image
    cinv2ds, areas = inverse_cov2d(cov2ds)

    image = splat(height, width, us, cinv2ds, gs['alpha'],
          depths, colors, areas, im)

    from PIL import Image
    pil_img = Image.fromarray((np.clip(image, 0, 1)*255).astype(np.uint8))
    pil_img.save('gs_image.png')

    #  plt.imshow(image)
    plt.show()
