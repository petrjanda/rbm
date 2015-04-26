package nn

import _root_.nn.conf.NNConf
import _root_.nn.fn.act.ActivationFunction
import _root_.nn.fn.loss.LossFunction
import nn.fn.learn.LearningFunction
import nn.trainer.{ContrastiveDivergence, RBMGradient}
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.sampling.Sampling._

object RBM {
  def apply(numVisible: Int, numHidden: Int, conf:NNConf)(implicit rng:MersenneTwister): RBM = {
    val v = Nd4j.zeros(numVisible)
    val h = Nd4j.zeros(numHidden)
    val W = normal(rng, Nd4j.zeros(numVisible, numHidden), .5)

    new RBM(W, v, h, conf)
  }
}

class RBM(val W: INDArray, val v: INDArray, val h: INDArray, conf:NNConf)(implicit rng:MersenneTwister) {
  lazy val numVisible = W.rows()

  lazy val numHidden = W.columns()

  /**
   * Loss function, given the input matrix.
   *
   * @param x Input matrix
   * @return The loss coeficient
   */
  def loss(x: INDArray): Double =
    math.abs(conf.loss(x, reconstruct(x), conf.lossReguralizer.l2, conf.lossReguralizer.use))

  /**
   * Reconstruct the given input.
   *
   * @param x Input matrix to be reconstructed
   * @return Output (reconstruction) matrix
   */
  def reconstruct(x: INDArray): INDArray =
    propDown(propUp(x))

  /**
   * Propagate the given input down.
   *
   * @param x
   * @return
   */
  private[nn] def propUp(x: INDArray): INDArray =
    conf.activation(x.mmul(W).addRowVector(h))

  /**
   * Propagate the given hidden layer value up.
   *
   * @param x
   * @return
   */
  private[nn] def propDown(x: INDArray): INDArray =
    conf.activation(x.mmul(W.transpose).addRowVector(v))

  /**
   * Update the RBM weights. As its immutable, it creates a new version of RBM with
   * new weights.
   *
   * @param grad RBM weight gradients
   * @return New RBM with updated weights
   */
  private[nn] def update(grad:RBMGradient) =
    new RBM(W.add(grad.W), v.add(grad.v), h.add(grad.h), conf)

  override def toString =
    s"RBM(W:$W, v:$v, h:$h)"
}
