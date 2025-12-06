import torch
from gsplat import rasterization
from dust3r.utils.geometry import inv, geotrf


def render(
    intrinsics: torch.Tensor,
    pts3d: torch.Tensor,
    rgbs: torch.Tensor | None = None,
    scale: float = 0.002,
    opacity: float = 0.95,
):

    device = pts3d.device
    batch_size = len(intrinsics)
    img_size = pts3d.shape[1:3]
    pts3d = pts3d.reshape(batch_size, -1, 3)
    num_pts = pts3d.shape[1]
    quats = torch.randn((num_pts, 4), device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = scale * torch.ones((num_pts, 3), device=device)
    opacities = opacity * torch.ones((num_pts), device=device)
    if rgbs is not None:
        assert rgbs.shape[1] == 3
        rgbs = rgbs.reshape(batch_size, 3, -1).transpose(1, 2)
    else:
        rgbs = torch.ones_like(pts3d[:, :, :3])

    rendered_rgbs = []
    rendered_depths = []
    accs = []
    for i in range(batch_size):
        rgbd, acc, _ = rasterization(
            pts3d[i],
            quats,
            scales,
            opacities,
            rgbs[i],
            torch.eye(4, device=device)[None],
            intrinsics[[i]],
            width=img_size[1],
            height=img_size[0],
            packed=False,
            render_mode="RGB+D",
        )

        rendered_depths.append(rgbd[..., 3])

    rendered_depths = torch.cat(rendered_depths, dim=0)

    return rendered_rgbs, rendered_depths, accs


def get_render_results(gts, preds, self_view=False):
    device = preds[0]["pts3d_in_self_view"].device
    with torch.no_grad():
        depths = []
        gt_depths = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if self_view:
                camera = inv(gt["camera_pose"]).to(device)
                intrinsics = gt["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_self_view"]
            else:
                camera = inv(gts[0]["camera_pose"]).to(device)
                intrinsics = gts[0]["camera_intrinsics"].to(device)
                pred = pred["pts3d_in_other_view"]
            gt_img = gt["img"].to(device)
            gt_pts3d = gt["pts3d"].to(device)

            _, depth, _ = render(intrinsics, pred, gt_img)
            _, gt_depth, _ = render(intrinsics, geotrf(camera, gt_pts3d), gt_img)
            depths.append(depth)
            gt_depths.append(gt_depth)
    return depths, gt_depths
