package nn

import com.typesafe._
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

  val trainer = RBMTrainer(
    epochs = 100,
    miniBatchSize = 10,
    learningRate = ConstantRate(.95)
  )

  it should "get reasonable loss after learning with ones as targets" in {
    val input = Nd4j.ones(20, 2)
    val dataSet = DataSet(input, input)
    val layer = MultiLayerRBM(2, List((1, Sigmoid)), CrossEntropy).layers.head
    val rbm = trainer.train(layer, dataSet)

    rbm.loss(input) should be < 0.01
  }

  it should "get reasonable loss after learning with zeros as targets" in {
    val input = Nd4j.zeros(20, 2)
    val dataSet = DataSet(input, input)
    val layer = RBM(2, 1, Sigmoid, CrossEntropy)
    val rbm = trainer.train(layer, dataSet)

    val nn = MultiLayerRBM(List(layer), CrossEntropy)

    rbm.loss(input) should be < 0.01
  }

  it should "calculate reasonable reconstruction for greedily trained multilayer net" in {
    val input = Nd4j.zeros(20, 2)
    val dataSet = DataSet(input, input)

    // Train layer #1
    val layer1 = trainer.train(RBM(2, 1, Sigmoid, CrossEntropy), dataSet)

    // Train layer #2
    val layer2 = trainer.train(RBM(1, 1, Sigmoid, CrossEntropy), DataSet(layer1.propUp(input), layer1.propUp(input)))


    // Build final multi-layer RBM
    val nn = MultiLayerRBM(List(layer1, layer2), CrossEntropy)

    nn.loss(input) should be < 0.01
  }
}