import nn.conf.{NNLossRegularizerConf, NNConf}
import nn.fn.act.Sigmoid
import nn.fn.loss.CrossEntropy
import nn.RBM
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.factory.Nd4j

object TestApp extends App {
  implicit val rng = new MersenneTwister(System.currentTimeMillis())
  val conf = NNConf(Sigmoid, CrossEntropy, NNLossRegularizerConf())
  val rbm = RBM(5, 5, conf)

  println(rbm.reconstruct(Nd4j.ones(5)))
}



