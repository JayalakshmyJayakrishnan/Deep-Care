class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torchvision.ops.misc.Conv2dNormActivation
  __annotations__["1"] = __torch__.torch.nn.modules.container.___torch_mangle_10.Sequential
  __annotations__["2"] = __torch__.torch.nn.modules.container.___torch_mangle_51.Sequential
  __annotations__["3"] = __torch__.torch.nn.modules.container.___torch_mangle_92.Sequential
  __annotations__["4"] = __torch__.torch.nn.modules.container.___torch_mangle_153.Sequential
  __annotations__["5"] = __torch__.torch.nn.modules.container.___torch_mangle_214.Sequential
  __annotations__["6"] = __torch__.torch.nn.modules.container.___torch_mangle_295.Sequential
  __annotations__["7"] = __torch__.torch.nn.modules.container.___torch_mangle_316.Sequential
  __annotations__["8"] = __torch__.torchvision.ops.misc.___torch_mangle_320.Conv2dNormActivation
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_321.Sequential,
    x: Tensor) -> Tensor:
    _8 = getattr(self, "8")
    _7 = getattr(self, "7")
    _6 = getattr(self, "6")
    _5 = getattr(self, "5")
    _4 = getattr(self, "4")
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _9 = (_2).forward((_1).forward((_0).forward(x, ), ), )
    _10 = (_5).forward((_4).forward((_3).forward(_9, ), ), )
    _11 = (_8).forward((_7).forward((_6).forward(_10, ), ), )
    return _11
