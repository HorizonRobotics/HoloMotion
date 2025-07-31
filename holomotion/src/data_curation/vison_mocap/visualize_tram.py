# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# -----------------------------------------------------------------------------
# Portions of this file are derived from tram (https://github.com/yufu-wang/tram).
# The original tram code is licensed under the MIT license.
# -----------------------------------------------------------------------------
import os
import sys

sys.path.append("../../thirdparties")
import torch

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import argparse
from glob import glob

import cv2
import imageio
import numpy as np
from joints2smpl import joints2smpl
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from tram.lib.models.smpl import SMPL
from tram.lib.vis.renderer import Renderer


def traj_filter(pred_vert_w, pred_j3d_w, sigma=3):
    """Smooth the root trajetory (xyz)."""
    root = pred_j3d_w[:, 0]
    root_smooth = torch.from_numpy(gaussian_filter(root, sigma=sigma, axes=0))

    pred_vert_w = pred_vert_w + (root_smooth - root)[:, None]
    pred_j3d_w = pred_j3d_w + (root_smooth - root)[:, None]
    return pred_vert_w, pred_j3d_w


def visualize_tram(
    seq_folder, floor_scale=2, bin_size=-1, max_faces_per_bin=30000
):
    """Visualize smpl human motion and generate amass-compatible npz.

    Reference:
        https://github.com/yufu-wang/tram/blob/main/lib/pipeline/visualization.py
    """
    img_folder = f"{seq_folder}/images"
    hps_folder = f"{seq_folder}/hps"
    imgfiles = sorted(glob(f"{img_folder}/*.jpg"))
    hps_files = sorted(glob(f"{hps_folder}/*.npy"))

    device = "cuda"
    smpl = SMPL().to(device)
    colors = np.loadtxt("data/colors.txt") / 255
    colors = torch.from_numpy(colors).float()

    max_track = len(hps_files)
    tstamp = [t for t in range(len(imgfiles))]
    track_verts = {i: [] for i in tstamp}
    track_joints = {i: [] for i in tstamp}
    track_tid = {i: [] for i in tstamp}
    locations = []
    lowest = []

    ##### TRAM + VIMO #####
    pred_cam = np.load(f"{seq_folder}/camera.npy", allow_pickle=True).item()
    img_focal = pred_cam["img_focal"].item()
    world_cam_r = torch.tensor(pred_cam["world_cam_R"]).to(device)
    world_cam_t = torch.tensor(pred_cam["world_cam_T"]).to(device)

    for i in range(max_track):
        hps_file = hps_files[i]
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl["pred_rotmat"].to(device)
        pred_shape = pred_smpl["pred_shape"].to(device)
        pred_trans = pred_smpl["pred_trans"].to(device)
        frame = pred_smpl["frame"]

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred = smpl(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=pred_shape,
            transl=pred_trans.squeeze(),
            pose2rot=False,
            default_smpl=True,
        )
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        cam_r = world_cam_r[frame]
        cam_t = world_cam_t[frame]

        pred_vert_w = (
            torch.einsum("bij,bnj->bni", cam_r, pred_vert) + cam_t[:, None]
        )
        pred_j3d_w = (
            torch.einsum("bij,bnj->bni", cam_r, pred_j3d) + cam_t[:, None]
        )
        pred_vert_w, pred_j3d_w = traj_filter(
            pred_vert_w.cpu(), pred_j3d_w.cpu()
        )
        locations.append(pred_j3d_w.mean(1))
        lowest.append(pred_vert_w[:, :, 1].min())

        for j, f in enumerate(frame.tolist()):
            track_tid[f].append(i)
            track_verts[f].append(pred_vert_w[j])
            track_joints[f].append(pred_j3d_w[j])

    offset = torch.min(torch.stack(lowest))
    locations = torch.cat(locations).to(device)
    for i in range(max_track):
        hps_file = hps_files[i]
        npz_file = hps_files[i].replace(".npy", ".npz")
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl["pred_rotmat"].to(device)
        pred_shape = pred_smpl["pred_shape"].to(device)
        pred_trans = pred_smpl["pred_trans"].to(device)
        frame = pred_smpl["frame"]

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred = smpl(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=pred_shape,
            transl=pred_trans.squeeze(),
            pose2rot=False,
            default_smpl=True,
        )
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        cam_r = world_cam_r[frame]
        cam_t = world_cam_t[frame]

        pred_vert_w = (
            torch.einsum("bij,bnj->bni", cam_r, pred_vert) + cam_t[:, None]
        )
        pred_j3d_w = (
            torch.einsum("bij,bnj->bni", cam_r, pred_j3d) + cam_t[:, None]
        )
        pred_vert_w, pred_j3d_w = traj_filter(
            pred_vert_w.cpu(), pred_j3d_w.cpu()
        )
        joint_lists = pred_j3d_w.numpy()
        joints2smpl(joint_lists, npz_file)

    offset = torch.tensor([0, offset, 0]).to(device)

    # locations = torch.cat(locations).to(device)
    cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
    sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * floor_scale

    ##### Viewing Camera #####
    world_cam_t = world_cam_t - offset
    view_cam_r = world_cam_r.mT.to("cuda")
    view_cam_t = -torch.einsum("bij,bj->bi", world_cam_r, world_cam_t).to(
        "cuda"
    )

    ##### Render video for visualization #####
    writer = imageio.get_writer(
        f"{seq_folder}/tram_output.mp4",
        fps=30,
        mode="I",
        format="FFMPEG",
        macro_block_size=1,
    )
    img = cv2.imread(imgfiles[0])
    renderer = Renderer(
        img.shape[1],
        img.shape[0],
        img_focal - 100,
        "cuda",
        smpl.faces,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    renderer.set_ground(scale, cx.item(), cz.item())

    for i in tqdm(range(len(imgfiles))):
        img = cv2.imread(imgfiles[i])[:, :, ::-1]

        verts_list = track_verts[i]
        if len(verts_list) > 0:
            verts_list = torch.stack(track_verts[i])[:, None].to("cuda")
            verts_list -= offset

            tid = track_tid[i]
            verts_colors = torch.stack([colors[t] for t in tid]).to("cuda")

        faces = renderer.faces.clone().squeeze(0)
        cameras, lights = renderer.create_camera_from_cv(
            view_cam_r[[i]], view_cam_t[[i]]
        )
        # cameras = PerspectiveCameras(T=view_cam_t[[i]])
        rend = renderer.render_with_ground_multiple(
            verts_list, faces, verts_colors, cameras, lights
        )

        out = np.concatenate([img, rend], axis=1)
        writer.append_data(out)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default="./example_video.mov", help="input video"
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=-1,
        help="rasterization bin_size; set to [64,128,...] to increase speed",
    )
    parser.add_argument(
        "--floor_scale", type=int, default=3, help="size of the floor"
    )
    args = parser.parse_args()

    # File and folders
    file = args.video
    root = os.path.dirname(file)
    seq = os.path.basename(file).split(".")[0]

    seq_folder = f"results/{seq}"
    img_folder = f"{seq_folder}/images"
    imgfiles = sorted(glob(f"{img_folder}/*.jpg"))

    ##### Combine camera & human motion #####
    # Render video
    print("Visualize results ...")
    visualize_tram(
        seq_folder, floor_scale=args.floor_scale, bin_size=args.bin_size
    )
