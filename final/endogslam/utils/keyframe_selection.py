import torch
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where((gt_depth[0] > 0) & (gt_depth[0] < 1e10)) # Remove invalid depth values
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices]

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]
        # Filter out the points that are outside the image
        edge = 20
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)
        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
    # Select the keyframes with percentage of points inside the image > 0
    selected_keyframe_list = [keyframe_dict['id']
                                for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:k])

    return selected_keyframe_list
    
    
def keyframe_selection_distance(time_idx, curr_position, keyframe_list, distance_current_frame_prob, n_samples):
    """
    Performs sampling based on a probability distribution and returns 
    the indices of `n_samples` selected keyframes.

    Args:
        probabilities: A list of probabilities representing the 
            likelihood of selecting each keyframe.
        n_samples: The number of keyframes to be sampled.

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """
    distances = []
    time_laps = []
    curr_shift = np.linalg.norm(curr_position)
    for keyframe in keyframe_list:
        est_w2c = keyframe['est_w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position)
        time_lap = time_idx - keyframe['id']
        distances.append(distance)
        time_laps.append(time_lap)

    def dis2prob(x, scaler):
        scaler = max(float(scaler), 1e-6)
        x = max(float(x), 0.0)
        return np.log2(1.0 + scaler / (x + scaler / 5.0))
    dis_prob = [dis2prob(d, curr_shift)+dis2prob(t, time_idx) for d, t in zip(distances, time_laps)]
    sum_prob = sum(dis_prob) / (1-distance_current_frame_prob) # distance_current_frame_prob： p_c
    norm_dis_prob = [p/sum_prob for p in dis_prob]
    norm_dis_prob.append(distance_current_frame_prob) # index 'len(keyframe_list)' indicate the current frame
    # Compute the cumulative distribution function (CDF).
    cdf = np.cumsum(norm_dis_prob)
    # Generate random samples.
    samples = np.random.rand(n_samples)
    # Select indices by comparing random numbers with CDF.
    sample_indices = np.searchsorted(cdf, samples)
    
    # no sampling ablation
    # sample_indices = []
    # for i in range(n_samples):
    #     # sample = np.random.randint(0, len(keyframe_list))
    #     sample_indices.append(len(keyframe_list))
        
    return sample_indices


def keyframe_distance_probs(time_idx, curr_position, keyframe_list):
    """
    Compute baseline distance-based probabilities for keyframes only (no current-frame token).
    Returns np.ndarray shape (N,), sum to 1.
    """
    n = len(keyframe_list)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    distances = []
    time_laps = []
    curr_shift = float(np.linalg.norm(curr_position))
    for keyframe in keyframe_list:
        est_w2c = keyframe['est_w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = float(np.linalg.norm(camera_position - curr_position))
        time_lap = float(time_idx - keyframe['id'])
        distances.append(distance)
        time_laps.append(time_lap)

    def dis2prob(x, scaler):
        scaler = max(float(scaler), 1e-6)
        x = max(float(x), 0.0)
        return np.log2(1.0 + scaler / (x + scaler / 5.0))
    raw = np.array(
        [dis2prob(d, curr_shift) + dis2prob(t, float(time_idx)) for d, t in zip(distances, time_laps)],
        dtype=np.float64,
    )
    raw = np.clip(raw, 1e-12, None)
    raw = raw / raw.sum()
    return raw


def keyframe_selection_scorenet(curr_frame_id, ref_frame_ids, scores, k):
    """
    Select top-k frames by ScoreNet scores.

    Args:
        curr_frame_id (int): current frame id (unused, reserved for future).
        ref_frame_ids (list[int]): candidate reference frame ids.
        scores (torch.Tensor or np.ndarray): (T,) scores aligned with ref_frame_ids.
        k (int): number of frames to select.
    Returns:
        selected_indices (list[int]): indices into ref_frame_ids.
    """
    if k <= 0 or len(ref_frame_ids) == 0:
        return []
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().view(-1)
        _, topk = torch.topk(scores, k=min(k, scores.numel()))
        return topk.tolist()
    scores = np.asarray(scores).reshape(-1)
    k = min(k, scores.shape[0])
    topk = np.argpartition(-scores, k-1)[:k]
    # sort by score desc for stability
    topk = topk[np.argsort(-scores[topk])]
    return topk.tolist()


def keyframe_sampling_scorenet(ref_frame_ids, scores, n_samples, temperature=1.0):
    """
    Sample keyframes according to ScoreNet weights.
    Returns a list of indices into ref_frame_ids with length n_samples.
    """
    if len(ref_frame_ids) == 0 or n_samples <= 0:
        return []
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().view(-1).numpy()
    scores = np.asarray(scores).reshape(-1)
    # softmax with temperature
    logits = scores / max(float(temperature), 1e-6)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / (probs.sum() + 1e-8)
    return np.random.choice(len(ref_frame_ids), size=n_samples, replace=True, p=probs).tolist()


def keyframe_sampling_scorenet_with_current(
    ref_frame_ids,
    scores,
    n_samples,
    current_frame_prob=0.1,
    temperature=1.0,
    deduplicate_keyframes=False,
    score_keep_ratio=1.0,
    temporal_min_gap=0,
):
    """
    ScoreNet sampling with explicit current-frame probability, matching baseline behavior.

    Returns indices in a mixed index space:
      - [0, len(ref_frame_ids)-1] for keyframe candidates
      - len(ref_frame_ids) indicates the current frame
    """
    if n_samples <= 0:
        return []
    if current_frame_prob < 0.0:
        current_frame_prob = 0.0
    if current_frame_prob > 1.0:
        current_frame_prob = 1.0

    current_token = len(ref_frame_ids)
    if len(ref_frame_ids) == 0:
        return [current_token] * n_samples

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().view(-1).numpy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep_ratio = float(np.clip(score_keep_ratio, 0.0, 1.0))
    if keep_ratio > 0.0 and keep_ratio < 1.0:
        keep_n = max(1, int(np.ceil(len(scores) * keep_ratio)))
        keep_idx = np.argpartition(-scores, keep_n - 1)[:keep_n]
        mask = np.zeros((len(scores),), dtype=bool)
        mask[keep_idx] = True
        logits = np.full_like(scores, -1e9, dtype=np.float64)
        logits[mask] = scores[mask]
    else:
        logits = scores
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - np.max(logits)
    probs = np.exp(np.clip(logits, -50.0, 50.0))
    probs = probs / (probs.sum() + 1e-12)

    out = []
    available = np.arange(len(ref_frame_ids), dtype=np.int64)
    temporal_min_gap = int(max(0, temporal_min_gap))
    for _ in range(n_samples):
        if np.random.rand() < current_frame_prob:
            out.append(current_token)
        else:
            use_available = (deduplicate_keyframes or temporal_min_gap > 0) and available.size > 0
            if use_available:
                p = probs[available]
                p = p / (p.sum() + 1e-12)
                chosen = int(np.random.choice(available, p=p))
            else:
                chosen = int(np.random.choice(len(ref_frame_ids), p=probs))
            out.append(chosen)

            if use_available:
                if temporal_min_gap > 0:
                    chosen_fid = int(ref_frame_ids[chosen])
                    gap_mask = np.array(
                        [abs(int(ref_frame_ids[idx]) - chosen_fid) >= temporal_min_gap for idx in available],
                        dtype=bool,
                    )
                else:
                    gap_mask = np.ones((available.size,), dtype=bool)
                if deduplicate_keyframes:
                    gap_mask = gap_mask & (available != chosen)
                available = available[gap_mask]
    return out


def keyframe_sampling_mixed_with_current(
    ref_frame_ids,
    scores,
    distance_probs,
    n_samples,
    current_frame_prob=0.1,
    alpha=0.5,
    temperature=1.0,
    deduplicate_keyframes=False,
    score_keep_ratio=1.0,
    temporal_min_gap=0,
):
    """
    Mixed sampling between baseline distance prior and ScoreNet posterior.
      mixed = alpha * distance_probs + (1-alpha) * scorenet_probs
    Returns indices in mixed index space:
      - [0, len(ref_frame_ids)-1] for keyframes
      - len(ref_frame_ids) for current frame
    """
    if n_samples <= 0:
        return []
    if current_frame_prob < 0.0:
        current_frame_prob = 0.0
    if current_frame_prob > 1.0:
        current_frame_prob = 1.0
    alpha = float(np.clip(alpha, 0.0, 1.0))

    current_token = len(ref_frame_ids)
    if len(ref_frame_ids) == 0:
        return [current_token] * n_samples

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().float().view(-1).numpy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep_ratio = float(np.clip(score_keep_ratio, 0.0, 1.0))
    if keep_ratio > 0.0 and keep_ratio < 1.0:
        keep_n = max(1, int(np.ceil(len(scores) * keep_ratio)))
        keep_idx = np.argpartition(-scores, keep_n - 1)[:keep_n]
        mask = np.zeros((len(scores),), dtype=bool)
        mask[keep_idx] = True
        logits = np.full_like(scores, -1e9, dtype=np.float64)
        logits[mask] = scores[mask]
    else:
        logits = scores
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - np.max(logits)
    scorenet_probs = np.exp(np.clip(logits, -50.0, 50.0))
    scorenet_probs = scorenet_probs / (scorenet_probs.sum() + 1e-12)

    distance_probs = np.asarray(distance_probs, dtype=np.float64).reshape(-1)
    if distance_probs.shape[0] != len(ref_frame_ids):
        distance_probs = np.ones((len(ref_frame_ids),), dtype=np.float64) / len(ref_frame_ids)
    else:
        distance_probs = np.clip(distance_probs, 1e-12, None)
        distance_probs = distance_probs / distance_probs.sum()

    mixed = alpha * distance_probs + (1.0 - alpha) * scorenet_probs
    mixed = np.clip(mixed, 1e-12, None)
    mixed = mixed / mixed.sum()

    out = []
    available = np.arange(len(ref_frame_ids), dtype=np.int64)
    temporal_min_gap = int(max(0, temporal_min_gap))
    for _ in range(n_samples):
        if np.random.rand() < current_frame_prob:
            out.append(current_token)
        else:
            use_available = (deduplicate_keyframes or temporal_min_gap > 0) and available.size > 0
            if use_available:
                p = mixed[available]
                p = p / (p.sum() + 1e-12)
                chosen = int(np.random.choice(available, p=p))
            else:
                chosen = int(np.random.choice(len(ref_frame_ids), p=mixed))
            out.append(chosen)

            if use_available:
                if temporal_min_gap > 0:
                    chosen_fid = int(ref_frame_ids[chosen])
                    gap_mask = np.array(
                        [abs(int(ref_frame_ids[idx]) - chosen_fid) >= temporal_min_gap for idx in available],
                        dtype=bool,
                    )
                else:
                    gap_mask = np.ones((available.size,), dtype=bool)
                if deduplicate_keyframes:
                    gap_mask = gap_mask & (available != chosen)
                available = available[gap_mask]
    return out
