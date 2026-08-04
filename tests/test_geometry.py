import numpy as np
from worldbridge.geometry import CameraModel


def test_camera_pixel_roundtrip_and_radial_depth():
    cam = CameraModel(128,128,35,32)
    uv = np.array([[0,0],[63,63],[127,127],[41,87]], np.float64)
    d = np.array([4.,7.,12.,9.], np.float64)
    world = cam.backproject_pixels(d, uv, np.zeros(3), np.array([1.,0.,0.,0.]))
    uv2, z, radial = cam.project(world, np.zeros(3), np.array([1.,0.,0.,0.]))
    np.testing.assert_allclose(uv2, uv, atol=1e-6)
    np.testing.assert_allclose(radial, d, atol=1e-6)
    assert (z > 0).all()


def test_quaternion_camera_translation_direction():
    cam = CameraModel(32,32,16,32)
    # 90 degrees around +Z: local +X maps to world +Y.
    q = np.array([np.cos(np.pi/4),0.,0.,np.sin(np.pi/4)])
    p = cam.backproject_pixels(np.array([5.]), np.array([[15.5,15.5]]), np.array([2.,3.,4.]), q)
    # Centre ray is camera -Z, hence world -Z after z rotation.
    np.testing.assert_allclose(p[0], [2.,3.,-1.], atol=1e-6)


def test_quaternion_matrix_is_orthonormal():
    q = np.array([[.7,.2,-.1,.6],[1.,0.,0.,0.]])
    r = CameraModel.quaternion_matrix(q)
    np.testing.assert_allclose(np.einsum("...ij,...kj->...ik",r,r), np.broadcast_to(np.eye(3),(2,3,3)), atol=1e-6)
