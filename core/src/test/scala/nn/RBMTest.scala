package nn

import _root_.ds.MNIST
import com.typesafe._
import nn.ds.DataSet
import nn.fn.act.Sigmoid
import nn.fn.learn.ConstantRate
import nn.fn.loss.CrossEntropy
import nn.trainer.RBMTrainer
import nn.util.Repo
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

  val repo = new Repo("src/test/resources/nets/")

  it should "get reasonable loss after learning with ones as targets" in {
    val mnist = MNIST.read("data/mnist/train-labels-idx1-ubyte", "data/mnist/train-images-idx3-ubyte", Some(1))
    val ds = DataSet(mnist._1.transpose)

    val l1 = trainer.train(RBM(784, 500, Sigmoid, Sigmoid, CrossEntropy), ds)

    repo.save(l1, "l1.o")

    val l2ds = DataSet(l1.propUp(ds.inputs))
    val l2 = trainer.train(RBM(500, 200, Sigmoid, Sigmoid, CrossEntropy), l2ds)

    val l3ds = DataSet(l2.propUp(l2ds.inputs))
    val l3 = trainer.train(RBM(200, 50, Sigmoid, Sigmoid, CrossEntropy), l3ds)

    val rbm = MultiLayerRBM(List(l1, l2, l3), CrossEntropy)

    val d = trainer.train(RBM(784, 50, Sigmoid, Sigmoid, CrossEntropy), ds)

    repo.save(rbm, "rbm-multi.o")
    repo.save(rbm, "rbm-single.o")

    println("deep: " + rbm.loss(ds.inputs))
    println("direct: " + d.loss(ds.inputs))
  }
}