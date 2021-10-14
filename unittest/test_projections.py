import torch
import aloscene
import numpy as np


def _test_disp_depth_points3d(depth, height, width, resize=True):

    assert torch.allclose(depth.as_tensor(), depth.as_disp().as_depth().as_tensor())

    if resize:
        assert torch.allclose(depth.as_tensor(), depth.resize((height * 2, width * 2)).resize((height, width)).as_tensor())

        assert torch.allclose(
            depth.as_tensor(),
            depth.resize((height * 2, width * 2)).resize((height, width)).as_disp().as_depth().as_tensor(),
        )

        assert torch.allclose(
            depth.as_tensor(),
            depth.resize((height * 2, width * 2)).as_disp().resize((height, width)).as_depth().as_tensor(),
        )

        assert torch.allclose(
            depth.as_tensor(),
            depth.as_disp().resize((height * 2, width * 2)).as_depth().resize((height, width)).as_tensor(),
        )

        assert torch.allclose(
            depth.as_tensor(),
            depth.as_disp().as_depth().as_points3d().as_depth(torch.zeros_like(depth), depth.cam_intrinsic).as_tensor(),
        )

    if resize:
        assert torch.allclose(
            depth.as_tensor(),
            depth.as_disp()
            .as_depth()
            .resize((height * 2, width * 2))
            .as_points3d()
            .as_depth(torch.zeros_like(depth), depth.cam_intrinsic)
            .as_tensor(),
        )

        assert torch.allclose(
            depth.as_tensor(),
            depth.as_disp()
            .as_depth()
            .resize((height / 2, width / 2))
            .as_points3d()
            .as_depth(depth, depth.cam_intrinsic)
            .as_tensor(),
        )


def test_disp_depth_points3d_projection1():
    np.random.seed(42)
    height = 480
    width = 640
    intrinsic = aloscene.CameraIntrinsic(focal_length=320, plane_size=(height, width))
    depth = aloscene.Depth(np.zeros((1, height, width)) + 10, baseline=0.25, camera_side="left")
    depth.append_cam_intrinsic(intrinsic)
    _test_disp_depth_points3d(depth, height, width)


def test_disp_depth_points3d_projection2():
    np.random.seed(42)
    height = 480
    width = 640

    intrinsic1 = aloscene.CameraIntrinsic(focal_length=320, plane_size=(height, width))
    depth1 = aloscene.Depth(np.zeros((1, height, width)) + 10, baseline=0.25, camera_side="left")
    depth1.append_cam_intrinsic(intrinsic1)
    intrinsic2 = aloscene.CameraIntrinsic(focal_length=128, plane_size=(height / 2, width / 2))
    depth2 = aloscene.Depth(np.zeros((1, height, width)) + 20, baseline=0.25, camera_side="left")
    depth2.append_cam_intrinsic(intrinsic2)

    depth = torch.cat([depth1.temporal(), depth2.temporal()], dim=0)
    _test_disp_depth_points3d(depth, height, width)


def test_disp_depth_points3d_projection3():
    np.random.seed(42)
    height = 480
    width = 640

    intrinsic1 = aloscene.CameraIntrinsic(focal_length=320, plane_size=(height, width))
    depth1 = aloscene.Depth(np.zeros((1, height, width)) + 10, baseline=0.25, camera_side="left")
    depth1.append_cam_intrinsic(intrinsic1)
    intrinsic2 = aloscene.CameraIntrinsic(focal_length=128, plane_size=(height / 2, width / 2))
    depth2 = aloscene.Depth(np.zeros((1, height, width)) + 20, baseline=0.25, camera_side="left")
    depth2.append_cam_intrinsic(intrinsic2)

    depth = torch.cat([depth1.batch(), depth2.batch()], dim=0)
    _test_disp_depth_points3d(depth, height, width)


def test_disp_depth_points3d_projection4():
    np.random.seed(42)
    height = 480
    width = 640

    intrinsic1 = aloscene.CameraIntrinsic(focal_length=320, plane_size=(height, width))
    depth1 = aloscene.Depth(np.zeros((1, height, width)) + 10, baseline=0.25, camera_side="left")
    depth1.append_cam_intrinsic(intrinsic1)
    depth1 = torch.cat([depth1.temporal(), depth1.temporal()], dim=0)
    depth1[-1] += 15

    intrinsic2 = aloscene.CameraIntrinsic(focal_length=128, plane_size=(height / 2, width / 2))
    depth2 = aloscene.Depth(np.zeros((1, height, width)) + 20, baseline=0.25, camera_side="left")
    depth2.append_cam_intrinsic(intrinsic2)
    depth2 = torch.cat([depth2.temporal(), depth2.temporal()], dim=0)
    depth2[-1] += 7

    depth = torch.cat([depth1.batch(), depth2.batch()], dim=0)

    # Resize not supported with temporal & batch dimension
    _test_disp_depth_points3d(depth, height, width, resize=False)


if __name__ == "__main__":
    test_disp_depth_points3d_projection1()
    test_disp_depth_points3d_projection2()
    test_disp_depth_points3d_projection4()
