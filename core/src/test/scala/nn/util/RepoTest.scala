package nn.util

import nn.RBM
import nn.fn.act.Sigmoid
import nn.fn.loss.CrossEntropy
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.{Matchers, FlatSpec}

class RepoTest extends FlatSpec with Matchers {
  it should "store configuration of the neural network" in {
    implicit val rng = new MersenneTwister(123)

    val repo = new Repo("src/test/resources/nets/")

    val rbm = RBM(10, 10, Sigmoid, Sigmoid, CrossEntropy)

    repo.save(rbm, "rbm.o")

    val loaded = repo.load[RBM]("rbm.o")

    loaded.h should equal(rbm.h)
    loaded.v should equal(rbm.v)
    loaded.W should equal(rbm.W)
    loaded.hiddenActivation should equal(rbm.hiddenActivation)
    loaded.visibleActivation should equal(rbm.visibleActivation)
  }
}
