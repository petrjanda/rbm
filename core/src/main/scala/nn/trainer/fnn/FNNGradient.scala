package nn.trainer.fnn

case class FNNGradient(W: INDArray) {
  def rate(r: Double) = FNNGradient(W.mul(r))
}
