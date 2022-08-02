import torch
from torch import nn
from torch.autograd import Function
try:
    import sort_vertices
    sort_vertices_error = None
except Exception as e:
    sort_vertices= None
    sort_vertices_error = e
    pass


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):

        if sort_vertices is None:
            raise Exception(f"To install: cd aloception/aloscene/utils/rotated_iou/cuda_op; python setup.py install --user. Error={sort_vertices_error}")

        idx = sort_vertices.sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


sort_v = SortVertices.apply

if __name__ == "__main__":
    import time

    v = torch.rand([8, 1024, 24, 2]).float().cuda()
    mean = torch.mean(v, dim=2, keepdim=True)
    v = v - mean
    m = (torch.rand([8, 1024, 24]) > 0.8).cuda()
    nv = torch.sum(m.int(), dim=-1).int().cuda()
    start = time.time()
    result = sort_v(v, m, nv)
    torch.cuda.synchronize()
    print("time: %.2f ms" % ((time.time() - start) * 1000))
    print(result.size())
    print(result[0, 0, :])
