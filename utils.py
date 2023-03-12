import math

import torch

EPS = 1e-7

class VonMisesFisher(torch.distributions.Distribution):
  """torch.distributions.Distribution implementation of a von Mises-Fisher."""

  arg_constraints = {
      "loc": torch.distributions.constraints.real,
      "scale": torch.distributions.constraints.positive,
  }
  support = torch.distributions.constraints.real
  has_rsample = True
  _mean_carrier_measure = 0

  def __init__(self, loc, scale, validate_args=None, k=1):
    self.dtype = loc.dtype
    self.loc = loc
    self.scale = scale
    self.device = loc.device
    self.__m = loc.shape[-1]
    self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(
        self.device)
    self.k = k

    super().__init__(self.loc.size(), validate_args=validate_args)

  def sample(self, shape=torch.Size()):
    with torch.no_grad():
      return self.rsample(shape)

  def rsample(self, shape=torch.Size()):
    shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

    w = (
        self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(
            shape=shape))

    v = (torch.distributions.Normal(
        torch.tensor(0, dtype=self.dtype).to(self.device),
        torch.tensor(1, dtype=self.dtype).to(self.device),
    ).sample(shape + torch.Size(self.loc.shape)).to(self.device).transpose(
        0, -1)[1:]).transpose(0, -1)
    v = v / v.norm(dim=-1, keepdim=True)

    w_ = torch.sqrt(torch.clamp(1 - (w**2), EPS))
    x = torch.cat((w, w_ * v), -1)
    z = self.__householder_rotation(x)

    return z.type(self.dtype)

  def __sample_w3(self, shape):
    shape = shape + torch.Size(self.scale.shape)
    u = (
        torch.distributions.Uniform(
            torch.tensor(0, dtype=self.dtype).to(self.device),
            torch.tensor(1, dtype=self.dtype).to(self.device),
        ).sample(shape).to(self.device))
    self.__w = (1 + torch.stack(
        [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) /
                self.scale)
    return self.__w

  def __sample_w_rej(self, shape):
    c = torch.sqrt((4 * (self.scale**2)) + (self.__m - 1)**2)
    b_true = (-2 * self.scale + c) / (self.__m - 1)

    # Using Taylor approximation with a smooth swift from 10 < scale < 11 to
    # avoid numerical errors for large scale.
    b_app = (self.__m - 1) / (4 * self.scale)
    s = torch.min(
        torch.max(
            torch.tensor([0.0], dtype=self.dtype, device=self.device),
            self.scale - 10,
        ),
        torch.tensor([1.0], dtype=self.dtype, device=self.device),
    )
    b = b_app * s + b_true * (1 - s)

    a = (self.__m - 1 + 2 * self.scale + c) / 4
    d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

    self.__b, (self.__e, self.__w) = b, self.__while_loop(
        b, a, d, shape, k=self.k)
    return self.__w

  @staticmethod
  def first_nonzero(x, dim, invalid_val=-1):
    mask = x > 0
    idx = torch.where(
        mask.any(dim=dim),
        mask.float().max(dim=1)[1].squeeze(),
        torch.tensor(invalid_val, device=x.device),
    )
    return idx

  def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
    # Matrix while loop: samples a matrix of [A, k] samples, to avoid looping
    # all together.
    b, a, d = [
        e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
        for e in (b, a, d)
    ]
    w, e, bool_mask = (
        torch.zeros_like(b).to(self.device),
        torch.zeros_like(b).to(self.device),
        (torch.ones_like(b) == 1).to(self.device),
    )

    sample_shape = torch.Size([b.shape[0], k])
    shape = shape + torch.Size(self.scale.shape)

    while bool_mask.sum() != 0:
      con1 = torch.tensor(
          (self.__m - 1) / 2, dtype=torch.float64).to(self.device)
      con2 = torch.tensor(
          (self.__m - 1) / 2, dtype=torch.float64).to(self.device)
      e_ = (
          torch.distributions.Beta(con1, con2).sample(sample_shape).to(
              self.device).type(self.dtype))

      u = (
          torch.distributions.Uniform(
              torch.tensor(0 + eps, dtype=self.dtype).to(self.device),
              torch.tensor(1 - eps, dtype=self.dtype).to(self.device),
          ).sample(sample_shape).to(self.device).type(self.dtype))

      w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
      t = (2 * a * b) / (1 - (1 - b) * e_)

      accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
      accept_idx = self.first_nonzero(
          accept, dim=-1, invalid_val=-1).unsqueeze(1)
      accept_idx_clamped = accept_idx.clamp(0)
      w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
      e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

      reject = accept_idx < 0
      accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

      w[bool_mask * accept] = w_[bool_mask * accept]
      e[bool_mask * accept] = e_[bool_mask * accept]

      bool_mask[bool_mask * accept] = reject[bool_mask * accept]

    return e.reshape(shape), w.reshape(shape)

  def __householder_rotation(self, x):
    u = self.__e1 - self.loc
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    z = x - 2 * (x * u).sum(-1, keepdim=True) * u
    return z