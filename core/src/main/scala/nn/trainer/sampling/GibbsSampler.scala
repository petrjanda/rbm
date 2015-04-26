package nn.trainer.sampling

import nn.RBM
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.sampling.Sampling._

object GibbsSampler {
  case class Sample(mean: INDArray, sample: INDArray)
  case class HVHSample(vhMean: INDArray, vhSample: INDArray, hvMean: INDArray, hvSample: INDArray)

  object HVHSample {
    def apply(vh:Sample, hv:Sample): HVHSample = HVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}

class GibbsSampler(rbm:RBM)(implicit rng:MersenneTwister) {
  import GibbsSampler._

  def sampleHiddenGivenVisible(v: INDArray) = {
    val mean = rbm.propUp(v)

    Sample(mean, binomial(mean, 1, rng))
  }

  def sampleVisibleGivenHidden(h: INDArray) = {
    val mean = rbm.propDown(h)

    Sample(mean, binomial(mean, 1, rng))
  }

  def sampleHiddenVisibleHidden(h: INDArray) = {
    val vh = sampleVisibleGivenHidden(h)
    val hv = sampleHiddenGivenVisible(vh.sample)

    HVHSample(vh, hv)
  }
}