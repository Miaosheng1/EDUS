
import numpy as np
import torch


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1,
                        keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1,
                        keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))

def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0),
                         view_selection_method='nearest',
                         view_selection_stride=None,
                         ):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    # num_select = 4
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(
            batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)

    if view_selection_method == 'nearest':
        if view_selection_stride is not None:
            idx = np.minimum(np.arange(1, num_select + 1, dtype=int)
                             * view_selection_stride, num_cams - 1)
            selected_ids = sorted_ids[idx]
        else:
            selected_ids = sorted_ids[:num_select]
    else:
        raise Exception('unknown view selection method!')

    return selected_ids

def find_corres_index(image_idx,indices):
    res = []
    for value in indices:
        index = torch.where(image_idx == value)[0]
        res.append(index.detach().cpu())
    res = torch.cat(res)
    return res


def get_source_images_from_current_imageid(image_batch,
                                           scene_id,
                                           image_id,
                                           all_pose, 
                                           num_select = 2,
                                           num_imgs_per_scene=0):
    assert num_imgs_per_scene > 0
    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    target_pose = all_pose[image_id]
    scene_id = scene_id[0]
    start_pose_id = num_imgs_per_scene * scene_id
    end_pose_id = num_imgs_per_scene* (scene_id+1)

    train_poses = all_pose[start_pose_id:end_pose_id,:,:]
    
    nearest_pose_ids = get_nearest_pose_ids(target_pose.detach().cpu().numpy(),
                         train_poses.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id=image_id % num_imgs_per_scene,
                         angular_dist_method='dist',
                         )
    
    nearest_pose_ids = nearest_pose_ids + scene_id.item()*num_imgs_per_scene
    # print(f"nearest_pose_ids: {nearest_pose_ids}")
    src_poses = all_pose[nearest_pose_ids,...]
    nearest_pose_ids = find_corres_index(image_batch['image_idx'],nearest_pose_ids)
    src_rgbs = image_batch['image'][nearest_pose_ids]
    if 'depth' in image_batch:
        src_depths = image_batch['depth'][nearest_pose_ids]  ## fix index 之后才可以检索图像
    else:
        src_depths = None
    
    src_poses = torch.cat([src_poses,all_pose[image_id].unsqueeze(dim=0)])
    return src_rgbs,src_poses,src_depths


def eval_source_images_from_current_imageid(image_batch,
                                           scene_id,
                                           eval_image_id,
                                           all_pose, 
                                           eval_poses,
                                           num_select = 2,
                                           num_imgs_per_scene=0):
    assert num_imgs_per_scene > 0
    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    eval_poses = torch.cat([eval_poses,eye[None,None,:].repeat(eval_poses.shape[0],1,1)],dim=1)
    target_pose = eval_poses[eval_image_id]
    start_pose_id = num_imgs_per_scene * scene_id
    end_pose_id = num_imgs_per_scene* (scene_id+1)
    train_poses = all_pose[start_pose_id:end_pose_id,:,:]
    
    nearest_pose_ids = get_nearest_pose_ids(target_pose.detach().cpu().numpy(),
                         train_poses.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id=-1,
                         angular_dist_method='dist',
                         )
    
    ## [2帧的情况下]drop90 选择 former frame  进行排序之后的 feature_concat 起来在eval效果更好
    # nearest_pose_ids = np.array(sorted(nearest_pose_ids)[:num_select])
    # nearest_pose_ids = np.array([nearest_pose_ids[0],nearest_pose_ids[0],nearest_pose_ids[0]])
    nearest_pose_ids = nearest_pose_ids + scene_id*num_imgs_per_scene
    src_poses = all_pose[nearest_pose_ids,...]
    nearest_pose_ids = find_corres_index(image_batch['image_idx'],nearest_pose_ids)
    src_rgbs = image_batch['image'][nearest_pose_ids]  ## fix index 之后才可以检索图像

    if 'depth' in image_batch:
        src_depths = image_batch['depth'][nearest_pose_ids]  ## fix index 之后才可以检索图像
    else:
        src_depths = None
    # for id,img in enumerate(src_rgbs):
    #     import imageio
    #     imageio.imwrite(f"{image_batch['image_idx'][nearest_pose_ids[id]]}_source.png", img.detach().cpu().numpy())
    #     depth_map = colormaps.apply_depth_colormap(src_depths[id].unsqueeze(-1))
    #     imageio.imwrite(f"{image_batch['image_idx'][nearest_pose_ids[id]]}_sdepth.png", depth_map.detach().cpu().numpy())
    
    # exit()
    src_poses = torch.cat([src_poses,target_pose.unsqueeze(dim=0)])
    return src_rgbs,src_poses,src_depths


def render_trajectory_source_pose(image_batch,
                                    scene_id,
                                    all_pose, 
                                    target_pose,
                                    num_select = 3,
                                    num_imgs_per_scene=0):
    assert num_imgs_per_scene > 0
    """(N,4,4) pose matrices"""
    eye = torch.tensor([0., 0., 0., 1.]).to(all_pose)
    all_pose = torch.cat([all_pose,eye[None,None,:].repeat(all_pose.shape[0],1,1)],dim=1)
    target_pose = torch.cat([target_pose,eye[None,:]],dim=0)
   
    start_pose_id = num_imgs_per_scene * scene_id
    end_pose_id = num_imgs_per_scene* (scene_id+1)
    train_poses = all_pose[start_pose_id:end_pose_id,:,:]
    
    nearest_pose_ids = get_nearest_pose_ids(target_pose.detach().cpu().numpy(),
                         train_poses.detach().cpu().numpy(),
                         num_select=num_select,
                         tar_id=-1,
                         angular_dist_method='dist',
                         )
    
    nearest_pose_ids = nearest_pose_ids + scene_id*num_imgs_per_scene
    src_poses = all_pose[nearest_pose_ids,...]
    nearest_pose_ids = find_corres_index(image_batch['image_idx'],nearest_pose_ids)
    src_rgbs = image_batch['image'][nearest_pose_ids]  ## fix index 之后才可以检索图像

    if 'depth' in image_batch:
        src_depths = image_batch['depth'][nearest_pose_ids]  ## fix index 之后才可以检索图像
    else:
        src_depths = None
    # for id,img in enumerate(src_rgbs):
    #     import imageio
    #     imageio.imwrite(f"{image_batch['image_idx'][nearest_pose_ids[id]]}_source.png", img.detach().cpu().numpy())
    #     depth_map = colormaps.apply_depth_colormap(src_depths[id].unsqueeze(-1))
    #     imageio.imwrite(f"{image_batch['image_idx'][nearest_pose_ids[id]]}_sdepth.png", depth_map.detach().cpu().numpy())
    
    # exit()
    src_poses = torch.cat([src_poses,target_pose.unsqueeze(dim=0)])
    return src_rgbs,src_poses,src_depths
