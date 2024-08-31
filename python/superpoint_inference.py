import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import torchvision.transforms

CELL_SIZE = 8

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # Shared Encoder.
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # Detector Head.
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # Descriptor Head.
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ Forward pass that jointly computes unprocessed point and descriptor
    tensors.
    Input
      x: Image pytorch tensor shaped N x 1 x H x W.
    Output
      semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
      desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    print(semi.shape)
    print(desc.shape)
    return semi, desc


class SuperPointFrontend(object):
  """ Wrapper around pytorch net to help with pre and post image processing. """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
    self.cell = CELL_SIZE # Size of each output cell. Keep this fixed.
    self.border_remove = 4 # Remove points this close to the border.

    # # Load the network in inference mode.
    # self.net = SuperPointNet()
    # if cuda:
    #   # Train on GPU, deploy on GPU.
    #   self.net.load_state_dict(torch.load(weights_path))
    #   self.net = self.net.cuda()
    # else:
    #   # Train on GPU, deploy on CPU.
    #   self.net.load_state_dict(torch.load(weights_path,
    #                            map_location=lambda storage, loc: storage))
    # self.net.eval()
    backend = 'qnnpack'
    torch.backends.quantized.engine = backend
    self.net = torch.load(weights_path)

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ Process a numpy image to extract points and descriptors.
    Input
      img - HxW numpy float32 input image in range [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc - 256xN numpy array of corresponding unit normalized descriptors.
      heatmap - HxW numpy heatmap in range [0,1] of point confidences.
      """
    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)

    unique_outs0 = torch.unique(outs[0])
    unique_outs1 = torch.unique(outs[1])

    scale0 = torch.min(unique_outs0[1:] - unique_outs0[:-1])
    scale1 = torch.min(unique_outs1[1:] - unique_outs1[:-1])

    q_outs0 = torch.round(outs[0] / scale0).int()
    q_outs1 = torch.round(outs[1] / scale1).int()

    return scale0, q_outs0, scale1, q_outs1, outs



    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    print(nodust.shape)
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    print(nodust.shape)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    print(heatmap.shape)
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    print(heatmap.shape)
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    print(heatmap.shape)
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap


class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)


if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='PyTorch SuperPoint Demo.')
  parser.add_argument('img0_path', type=str, default='../datasets/kitti/sequences/00/image_0/000000.png',
      help='Path to input image0.')
  parser.add_argument('img1_path', type=str, default='../datasets/kitti/sequences/00/image_0/000000.png',
      help='Path to input image1.')
  parser.add_argument('output_path', type=str, default='',
      help='Path to output file.')
  parser.add_argument('gt_path', type=str, default='',
      help='Path to gt file.')
  parser.add_argument('--weights_path', type=str, default='superpoint_quantized_nonorm.pt',
      help='Path to pretrained weights file (default: superpoint_quantized_nonorm.pt).')
  parser.add_argument('--max_features', type=int, default=None,
      help='Maximum number of keypoints to print (default: no limit).')
  parser.add_argument('--nms_dist', type=int, default=4,
      help='Non Maximum Suppression (NMS) distance (default: 4).')
  parser.add_argument('--conf_thresh', type=float, default=0.015,
      help='Detector confidence threshold (default: 0.015).')
  parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
  parser.add_argument('--cuda', action='store_true',
      help='Use cuda GPU to speed up network processing speed (default: False)')
  opt = parser.parse_args()
  print(opt)

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = SuperPointFrontend(weights_path=opt.weights_path,
                          nms_dist=opt.nms_dist,
                          conf_thresh=opt.conf_thresh,
                          nn_thresh=opt.nn_thresh,
                          cuda=opt.cuda)
  print('==> Successfully loaded pre-trained network.')

  # Load image from img_path
  img0 = cv2.imread(opt.img0_path, 0).astype('float32') / 255.0
  img1 = cv2.imread(opt.img1_path, 0).astype('float32') / 255.0

  img0 = torch.from_numpy(np.expand_dims(img0, axis=0))
  img1 = torch.from_numpy(np.expand_dims(img1, axis=0))

  # # Step 1: Crop the image to (330, 880) centered
  # crop_transform = torchvision.transforms.CenterCrop((330, 880))
  # img0 = crop_transform(img0)
  # img1 = crop_transform(img1)

  # Step 2: Resize the cropped image to (, 240)
  resize_transform = torchvision.transforms.Resize((192, 640))
  img0 = resize_transform(img0)
  img1 = resize_transform(img1)

  img0 = torch.squeeze(img0, dim=0).numpy()
  img1 = torch.squeeze(img1, dim=0).numpy()

  # crop_shape = (640, 192)
  # img0 = cv2.resize(img0, crop_shape)
  # img1 = cv2.resize(img1, crop_shape)

  print("image size = ", img0.shape)

  semi_scale = [0, 0]
  semi = [None, None]
  desc_scale = [0, 0]
  desc = [None, None]
  outs = [None, None]
  img = [img0, img1]

  semi_scale[0], semi[0], desc_scale[0], desc[0], outs[0] = fe.run(img0)
  semi_scale[1], semi[1], desc_scale[1], desc[1], outs[1] = fe.run(img1)

  with open(opt.output_path, 'w') as fout:
      fout.write("#pragma once\n\n")
      fout.write("#include <stdint.h>\n\n")

      cell_size = CELL_SIZE
      fout.write(f"const int cell_size = {cell_size};\n\n")

      for i in [0, 1]:
          h, w = img[i].shape
          hc, wc = h // cell_size, w // cell_size
          prefix = f"image{i}"
          fout.write(f"const int {prefix}_rows = {img[i].shape[0]};\n")
          fout.write(f"const int {prefix}_cols = {img[i].shape[1]};\n")
          fout.write(f"const int {prefix}_channels = 1;\n\n")

          fout.write(f"const int {prefix}_feature_rows = {img[i].shape[0] // cell_size};\n")
          fout.write(f"const int {prefix}_feature_cols = {img[i].shape[1] // cell_size};\n\n")

          fout.write(f"const float {prefix}_semi_scale = {semi_scale[0]};\n")
          fout.write(f"const int8_t {prefix}_semi[{wc * hc}][65] = {{\n")
          for c in range(wc):
              for r in range(hc):
                  for cc in range(65):
                      fout.write(f"{semi[i][0, cc, r, c]}, ")
                  fout.write("\n")
          fout.write("};\n\n")

          fout.write(f"const float {prefix}_desc_scale = {desc_scale[0]};\n")
          fout.write(f"const int8_t {prefix}_desc[{wc * hc}][256] = {{\n")
          for c in range(wc):
              for r in range(hc):
                  for cc in range(256):
                      fout.write(f"{desc[i][0, cc, r, c]}, ")
                  fout.write("\n")
          fout.write("};\n\n")

  if opt.gt_path.strip():
      with open(opt.gt_path, 'w') as fout:
          fout.write("#pragma once\n\n")
          fout.write("#include <stdint.h>\n\n")

          cell_size = CELL_SIZE
          fout.write(f"const int cell_size = {cell_size};\n\n")
          for i in [0, 1]:
              semi, coarse_desc = outs[i][0], outs[i][1]
              # Convert pytorch -> numpy.
              semi = semi.data.cpu().numpy().squeeze()
              # --- Process points.
              dense = np.exp(semi) # Softmax.
              dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.

              dense = dense[:-1,:,:]

              print(dense.shape)

              max_vals = np.max(dense, axis=0)
              max_indices = np.argmax(dense, axis=0)

              print(max_vals.shape)

              h, w = img[i].shape
              hc, wc = h // cell_size, w // cell_size
              prefix = f"image{i}"
              fout.write(f"const int {prefix}_rows_gt = {img[i].shape[0]};\n")
              fout.write(f"const int {prefix}_cols_gt = {img[i].shape[1]};\n")
              fout.write(f"const int {prefix}_channels_gt = 1;\n\n")


              fout.write(f"const int {prefix}_feature_rows_gt = {img[i].shape[0] // cell_size};\n")
              fout.write(f"const int {prefix}_feature_cols_gt = {img[i].shape[1] // cell_size};\n\n")

              fout.write(f"const float {prefix}_probs_gt[{wc}][{hc}] = {{\n")
              for c in range(wc):
                  for r in range(hc):
                      fout.write(f"{max_vals[r, c]},\n")
              fout.write("};\n\n")

              fout.write(f"const int {prefix}_indices_gt[{wc}][{hc}] = {{\n")
              for c in range(wc):
                  for r in range(hc):
                      fout.write(f"{max_indices[r, c]},\n")
              fout.write("};\n\n")


  exit(0)


  num_features0 = pts0.shape[1] if opt.max_features is None else min(opt.max_features, pts0.shape[1])
  num_features1 = pts1.shape[1] if opt.max_features is None else min(opt.max_features, pts1.shape[1])

  prefix0 = "image0"
  prefix1 = "image1"

  with open(opt.output_path, 'w') as fout:
      # Write header file
      fout.write("#pragma once\n\n")
      fout.write("#include <stdint.h>\n\n")

      cell_size = CELL_SIZE
      feature_rows0 = heatmap0.shape[0] // cell_size
      feature_cols0 = heatmap0.shape[1] // cell_size

      fout.write(f"const int {prefix0}_rows = {img0.shape[0]};\n")
      fout.write(f"const int {prefix0}_cols = {img0.shape[1]};\n")
      fout.write(f"const int {prefix0}_channels = 1;\n\n")

      fout.write(f"const int {prefix0}_feature_rows = {feature_rows0};\n")
      fout.write(f"const int {prefix0}_feature_cols = {feature_cols0};\n")
      fout.write(f"const int {prefix0}_num_features = {num_features0};\n")
      fout.write(f"const int {prefix0}_feature_xs[{num_features0}] = {{\n")
      for i in range(num_features0):
          fout.write(f"{int(pts0[0, i])}, ")
      fout.write("};\n\n")
      fout.write(f"const int {prefix0}_feature_ys[{num_features0}] = {{\n")
      for i in range(num_features0):
          fout.write(f"{int(pts0[1, i])}, ")
      fout.write("};\n\n")
      fout.write(f"const float {prefix0}_feature_scores[{num_features0}] = {{\n")
      for i in range(num_features0):
          fout.write(f"{pts0[2, i]:.6f}, ")
      fout.write("};\n\n")
      fout.write(f"const float {prefix0}_feature_descriptors[{num_features0}][256] = {{\n")
      for i in range(num_features0):
          fout.write("{")
          for j in range(256):
              fout.write(f"{desc0[j, i]:.6f}, ")
          fout.write("},\n")
      fout.write("};\n\n")

      # Calculate the coordinate to index mapping
      # Initialize the mapping with -1
      coord_to_index0 = np.full((feature_rows0, feature_cols0), -1, dtype=np.int32)
      for i in range(num_features0):
          x = int(pts0[0, i])
          y = int(pts0[1, i])
          coord_to_index0[y // cell_size, x // cell_size] = i

      # Write the mapping to the header file
      fout.write(f"const int {prefix0}_coord_to_index[{feature_rows0 * feature_cols0}] = {{\n")
      for i in range(feature_rows0):
          for j in range(feature_cols0):
              fout.write(f"{coord_to_index0[i, j]}, ")
          fout.write("\n")
      fout.write("};\n\n")

      feature_rows1 = heatmap1.shape[0] // cell_size
      feature_cols1 = heatmap1.shape[1] // cell_size

      fout.write(f"const int {prefix1}_rows = {img1.shape[0]};\n")
      fout.write(f"const int {prefix1}_cols = {img1.shape[1]};\n")
      fout.write(f"const int {prefix1}_channels = 1;\n\n")
      fout.write(f"const int {prefix1}_feature_rows = {feature_rows1};\n")
      fout.write(f"const int {prefix1}_feature_cols = {feature_cols1};\n")
      fout.write(f"const int {prefix1}_num_features = {num_features1};\n")
      fout.write(f"const int {prefix1}_feature_xs[{num_features1}] = {{\n")
      for i in range(num_features1):
          fout.write(f"{int(pts1[0, i])}, ")
      fout.write("};\n\n")
      fout.write(f"const int {prefix1}_feature_ys[{num_features1}] = {{\n")
      for i in range(num_features1):
          fout.write(f"{int(pts1[1, i])}, ")
      fout.write("};\n\n")
      fout.write(f"const float {prefix1}_feature_scores[{num_features1}] = {{\n")
      for i in range(num_features1):
          fout.write(f"{pts1[2, i]:.6f}, ")
      fout.write("};\n\n")
      fout.write(f"const float {prefix1}_feature_descriptors[{num_features1}][256] = {{\n")
      for i in range(num_features1):
          fout.write("{")
          for j in range(256):
              fout.write(f"{desc1[j, i]:.6f}, ")
          fout.write("},\n")
      fout.write("};\n\n")

      # Calculate the coordinate to index mapping
      # Initialize the mapping with -1
      coord_to_index1 = np.full((feature_rows1, feature_cols1), -1, dtype=np.int32)
      for i in range(num_features1):
          x = int(pts1[0, i])
          y = int(pts1[1, i])
          coord_to_index1[y // cell_size, x // cell_size] = i
      
      # Write the mapping to the header file
      fout.write(f"const int {prefix1}_coord_to_index[{feature_rows1 * feature_cols1}] = {{\n")
      for i in range(feature_rows1):
          for j in range(feature_cols1):
              fout.write(f"{coord_to_index1[i, j]}, ")
          fout.write("\n")
      fout.write("};\n\n")




