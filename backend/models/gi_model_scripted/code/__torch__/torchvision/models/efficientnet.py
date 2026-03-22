class EfficientNet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  features : __torch__.torch.nn.modules.container.___torch_mangle_321.Sequential
  avgpool : __torch__.torch.nn.modules.pooling.___torch_mangle_322.AdaptiveAvgPool2d
  classifier : __torch__.torch.nn.modules.container.___torch_mangle_323.Sequential
  def forward(self: __torch__.torchvision.models.efficientnet.EfficientNet,
    x: Tensor) -> Tensor:
    classifier = self.classifier
    avgpool = self.avgpool
    features = self.features
    _0 = (avgpool).forward((features).forward(x, ), )
    input = torch.flatten(_0, 1)
    return (classifier).forward(input, )
class MBConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  block : __torch__.torch.nn.modules.container.Sequential
  stochastic_depth : __torch__.torchvision.ops.stochastic_depth.StochasticDepth
  def forward(self: __torch__.torchvision.models.efficientnet.MBConv,
    argument_1: Tensor) -> Tensor:
    block = self.block
    return (block).forward(argument_1, )
