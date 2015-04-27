package nn

import com.typesafe._
import nn.conf.NNConf
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

class RBMTest extends FlatSpec with Matchers {
  implicit val logger = scalalogging.Logger(LoggerFactory.getLogger("test"))
  
  implicit val rng = new MersenneTwister(123)

  val conf = NNConf(
    activation = Sigmoid,
    loss = CrossEntropy
  )

  val trainer = RBMTrainer(
    epochs = 10,
    miniBatchSize = 10,
    learningRate = ConstantRate(.95)
  )

  it should "get reasonable loss after 10 learning iterations with ones as targets" in {
    val input = Nd4j.ones(20, 2)
    val dataSet = DataSet(input, input)
    val rbm = trainer.train(RBM(2, 1, conf), dataSet)

    rbm.loss(input) should be < 0.01
  }

  it should "get reasonable loss after 10 learning iterations with zeros as targets" in {
    val input = Nd4j.zeros(20, 2)
    val dataSet = DataSet(input, input)
    val rbm = trainer.train(RBM(2, 1, conf), dataSet)

    rbm.loss(input) should be < 0.01
  }
}