package nn

import nn.conf.NNConf
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}

class RBMTest extends FlatSpec with Matchers {
  it should "get reasonable loss after 10 learning iterations" in {
    val input = Nd4j.ones(20, 2)
    val dataSet = DataSet(input, input)

    implicit val rng = new MersenneTwister(123)

    val conf = NNConf(
      activation = Sigmoid,
      loss = CrossEntropy
    )

    val rbm = RBMTrainer(
      epochs = 10000,
      miniBatchSize = 10,
      learningRate = ConstantRate(.95)
    ).train(RBM(2, 1, conf), dataSet)

    rbm.loss(input) should be < 0.01
  }
}