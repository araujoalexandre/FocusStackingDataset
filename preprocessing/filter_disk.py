import torch
import numpy as np
from numpy import arctan, sqrt, abs



class FilterDisk:

  def __init__(self):
    self._disks = {}

  def _filter_disk(self, radius):

    if radius == 0:
      return np.array([[1]])

    ax = radius
    corner = np.floor(radius / sqrt(2) + 0.5) - 0.5
    rsq = radius**2
    ## First set values for points completely covered by the disk
    X, Y = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    rhi = (abs(X) + 0.5)**2 + (abs(Y) + 0.5)**2
    f = (rhi <= rsq) / 1.0
    xx = np.linspace(0.5, radius - 0.5, radius)
    ii = sqrt(rsq - xx**2) # intersection points for sqrt (r^2 - x^2)
    ## Set the values at the axis caps
    tmp = sqrt(rsq - 0.25)
    rint = (0.5*tmp + rsq * arctan(0.5 / tmp)) / 2 # value of integral on the right
    cap = 2*rint - radius + 0.5 # at the caps, lint = rint
    f[ax       , ax+radius] = cap
    f[ax       , ax-radius] = cap
    f[ax+radius, ax       ] = cap
    f[ax-radius, ax       ] = cap
    if radius == 1:
      y = ii[0]
      lint = rint
      tmp = sqrt(rsq - y**2)
      rint = (y * tmp + rsq * arctan(y / tmp)) / 2
      val  = rint - lint - 0.5 * (y - 0.5)
      f[ax-radius, ax-radius] = val
      f[ax+radius, ax-radius] = val
      f[ax-radius, ax+radius] = val
      f[ax+radius, ax+radius] = val
    else:
      ## Set the values elsewhere on the rim
      idx = 0 # index in the vector ii
      x   = 0.5 # bottom left corner of the current square
      y   = radius - 0.5
      rx  = 0.5 # x on the right of the integrable region
      ybreak = False # did we change our y last time
      while True:
        i = x + 0.5
        j = y + 0.5
        lint = rint
        lx = rx
        if ybreak:
          ybreak = False
          val = lx - x
          idx += 1
          x += 1
          rx = x
          val -= y * (x - lx)
        elif ii[idx + 1] < y:
          ybreak = True
          y -= 1
          rx  = ii[int(y + 1.5)-1]
          val = (y + 1) * (x - rx)
        else:
          val = -y;
          idx += 1
          x += 1
          rx = x
          if np.floor(ii[idx] - 0.5) == y:
            y += 1
        tmp  = sqrt(rsq - rx**2)
        rint = (rx * tmp + rsq * arctan(rx / tmp)) / 2
        val += rint - lint
        f[int(ax + i), int(ax + j)] = val
        f[int(ax + i), int(ax - j)] = val
        f[int(ax - i), int(ax + j)] = val
        f[int(ax - i), int(ax - j)] = val
        f[int(ax + j), int(ax + i)] = val
        f[int(ax + j), int(ax - i)] = val
        f[int(ax - j), int(ax + i)] = val
        f[int(ax - j), int(ax - i)] = val
        if y < corner or x > corner:
          break

    # f /= np.pi * rsq
    return f
      
  def __call__(self, radius, size=None):
    radius = round(radius, 2)
    if radius in self._disks.keys():
      return self._disks[radius]
    p = 0
    if size is not None and size > (2*np.ceil(radius)+1):
      p = int((size - (2*np.ceil(radius)+1)) // 2)
    if int(radius) == radius:
      f = self._filter_disk(int(radius))
      f = f / f.sum()
      k = torch.FloatTensor(np.pad(f, ((p, p), (p, p))))
      self._disks['{:.2f}'.format(radius)] = k
      return k
    diff = radius - int(radius)
    R1 = self._filter_disk(int(np.ceil(radius)))
    R2 = np.pad(self._filter_disk(int(np.floor(radius))), ((1, 1), (1, 1)))
    f = diff * (R1 - R2) + R2
    f = f / f.sum()
    k = torch.FloatTensor(np.pad(f, ((p, p), (p, p))))
    self._disks['{:.2f}'.format(radius)] = k
    return k



if __name__ == '__main__':

  import pickle

  def pickle_dump(file, path):
    """function to dump picke object."""
    with open(path, 'wb') as f:
      pickle.dump(file, f, -1)

  filter_disk = FilterDisk()

  for r in np.arange(0, 41, 0.01):
    if r in np.arange(0, 41):
      print(r)
    filter_disk(r, size=41)

  pickle_dump(filter_disk._disks, 'filter_disk.pkl')

