class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torchvision.models.efficientnet.MBConv
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_10.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "0")
    return (_0).forward(argument_1, )
