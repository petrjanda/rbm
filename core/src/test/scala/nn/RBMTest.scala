package nn

import _root_.ds.MNIST
import com.typesafe._
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

class RBMTest extends FlatSpec with Matchers {
  implicit val logger = scalalogging.Logger(LoggerFactory.getLogger("test"))
  
  implicit val rng = new MersenneTwister(123)

  val trainer = RBMTrainer(
    epochs = 1,
    miniBatchSize = 25,
    learningRate = ConstantRate(0.05)
  )

  it should "get reasonable loss after learning with ones as targets" in {
    val mnist = MNIST.read("data/mnist/train-labels-idx1-ubyte", "data/mnist/train-images-idx3-ubyte", Some(1000))
    val ds = DataSet(mnist._1.transpose)

    val l1 = trainer.train(RBM(784, 500, Sigmoid, Sigmoid, CrossEntropy), ds)
    val l2 = trainer.train(RBM(500, 200, Sigmoid, Sigmoid, CrossEntropy), DataSet(l1.propUp(ds.inputs)))
    val rbm = MultiLayerRBM(List(l1, l2), CrossEntropy)

    val d = trainer.train(RBM(784, 200, Sigmoid, Sigmoid, CrossEntropy), ds)

    println("rbm: " + rbm.loss(ds.inputs))
    println("d: " + d.loss(ds.inputs))
  }
}