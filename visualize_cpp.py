import numpy as np
import torch
import os
import ctypes

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'library/build/libvisualize.so'))
lib.createViewer()


def clear_viewer():
    lib.clearViewer()


def remove_cloud(name=None):
    if name is None:
        lib.reomveAllCloud()
    else:
        lib.removeCloud(ctypes.create_string_buffer(name.encode('utf-8')))


def remove_shape(name=None):
    if name is None:
        lib.removeAllGrasps()
    else:
        lib.reomveShape(name)


def show():
    lib.runViewer()


def plot_marker():
    lib.plotMarker()


def type_convert(data, data_type, list_shape_assertion_fn=None):
    if isinstance(data, torch.Tensor):
        data = data.clone().detach().contiguous().cpu().numpy().astype(data_type)
    elif isinstance(data, np.ndarray):
        data = data.copy().astype(data_type)
    elif isinstance(data, list):
        data = np.array(data, dtype=data_type)
        if list_shape_assertion_fn is not None:
            assert list_shape_assertion_fn(data.shape)
    else:
        raise ValueError
    return data


def plot_cloud(cloud, color=None, name='cloud'):
    cloud = type_convert(cloud, np.float32, lambda s: len(s) == 2 and s[1] == 3)
    c_name = ctypes.create_string_buffer(name.encode('utf-8'))

    if color is not None:
        color = type_convert(color, np.int32)

        if len(color.shape) == 1:
            assert color.shape[0] == 3
            lib.plotCloud(cloud.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          int(cloud.shape[0]), int(color[0]), int(color[1]), int(color[2]), c_name)
        elif len(color.shape) == 2:
            assert color.shape[0] == cloud.shape[0] and color.shape[1] == 3
            lib.plotCloudWithColor(cloud.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   color.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                   int(cloud.shape[0]), c_name)
        else:
            raise ValueError
    else:
        lib.plotCloud(cloud.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), cloud.shape[0], 0, 0, 255, c_name)


def plot_grasps(grasps, color=None):
    grasps = type_convert(grasps, np.float32, lambda s: len(s) == 3 and s[1:] == (4, 4))

    if color is None:
        lib.plotGrasps(grasps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), grasps.shape[0], 0, 255, 0)
    else:
        color = type_convert(color, np.int32)
        if len(color.shape) == 1:
            assert color.shape[0] == 3
            lib.plotGrasps(grasps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           grasps.shape[0], color[0].item(), color[1].item(), color[2].item())
        elif len(color.shape) == 2:
            assert color.shape[0] == grasps.shape[0] and color.shape[1] == 3
            lib.plotGraspsWithColor(grasps.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                    color.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), grasps.shape[0])


def plot_line(p1, p2, color=None, name='line'):
    p1 = type_convert(p1, np.float32, lambda s: s == (3,)).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p2 = type_convert(p2, np.float32, lambda s: s == (3,)).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if color is not None:
        color = type_convert(color, np.int32, lambda s: s == (3,)).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    else:
        color = np.array([0, 0, 255], dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    lib.plotLine(p1, p2, color, ctypes.create_string_buffer(name.encode('utf-8')))
