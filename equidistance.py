import collections
import cv2
import numpy as np
import copy
from functools import lru_cache


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)


def convert_to_equidistant_image(image, source_camera_model) -> np.array:
    assert(source_camera_model.model == 'OPENCV_FISHEYE')

    # Convert to Equidistant fisheye model
    target_camera_model = source_camera_model
    target_camera_model = Camera(
        id=0,
        model = source_camera_model.model,
        width = source_camera_model.width,
        height = source_camera_model.height,
        params = np.zeros(target_camera_model.params.shape)
    )
    # Copy only fx, fy, cx, cy, and dicard distortion coeffs because equidistant model does not have them.
    target_camera_model.params[:4] = source_camera_model.params[:4]

    image = do_convert_image(image, source_camera_model, target_camera_model)

    return image, target_camera_model

def do_convert_image(source_image, source_camera_model, target_camera_model):
    uvs = gen_map(source_camera_model, target_camera_model)
    target_image = cv2.remap(source_image, uvs[..., 0], uvs[..., 1], interpolation=cv2.INTER_LINEAR)
    return target_image


def gen_map(source_camera_model, target_camera_model):
    width = source_camera_model.width
    height = source_camera_model.height

    fxs, fys, cxs, cys = source_camera_model.params[:4]
    fxt, fyt, cxt, cyt = target_camera_model.params[:4]
    Ds = tuple(source_camera_model.params[4:])
    Dt = tuple(target_camera_model.params[4:])

    Ks = tuple(((fxs, 0.0, cxs), (0.0, fys, cys), (0.0, 0.0, 1.0)))
    Kt = tuple(((fxt, 0.0, cxt), (0.0, fyt, cyt), (0.0, 0.0, 1.0)))

    return do_gen_map((Ds, Ks), (Dt, Kt), width, height)


@lru_cache
def do_gen_map(source_params, target_params, width, height):
    Ds, Ks = source_params
    Dt, Kt = target_params
    u, v = np.meshgrid(np.arange(0, width, dtype=np.float32), np.arange(0, height, dtype=np.float32))
    uvt = np.stack([u, v], axis=-1)
    undistorted = cv2.fisheye.undistortPoints(uvt, np.array(Kt), np.array(Dt))
    undistorted = undistorted
    return cv2.fisheye.distortPoints(undistorted, np.array(Ks), np.array(Ds))


def get_sample_camera_model_and_image():
    image = cv2.imread("../data/living_fisheye2/success/images/image_00001.png")    
    source_camera_model = Camera(id=1, model='OPENCV_FISHEYE', width=1620, height=1080,
                              params=np.array([ 5.36873726e+02,  5.45881952e+02,  7.72274701e+02,  5.70276609e+02, -3.82787817e-02, -4.22521054e-03, -1.29435232e-03,  1.60057318e-04]))
    
    return image, source_camera_model


def show_image(image):
    wh = image.shape[1], image.shape[0]
    wh_resized = (int(wh[0]/2), int(wh[1]/2))
    cv2.imshow("image2", cv2.resize(image, wh_resized))
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image, source_camera_model = get_sample_camera_model_and_image()
    show_image(image)
    image, target_camera_model = convert_to_equidistant_image(image, source_camera_model)
    mask = cv2.imread("../data/kerare_mask.png", 1)
    image[mask == 0] = 0
    print("target_camera_model: ", target_camera_model)
    show_image(image)
