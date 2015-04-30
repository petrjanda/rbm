package nn.trainer.sampling

import nn.RBM
import nn.fn.act.Sigmoid
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.sampling.Sampling._

object GibbsSampler {
  case class Sample(mean: INDArray, sample: INDArray)
  case class HVHSample(nvMean: INDArray, nvSamples: INDArray, nhMeans: INDArray, nhSample: INDArray)

  object HVHSample {
    def apply(vh:Sample, hv:Sample): HVHSample = HVHSample(vh.mean, vh.sample, hv.mean, hv.sample)
  }
}

class GibbsSampler(rbm:RBM)(implicit rng:MersenneTwister) {
  val isBinomialVisible = rbm.visibleActivation == Sigmoid
  val isBinomialHidden = rbm.hiddenActivation == Sigmoid

  import nn.trainer.sampling.GibbsSampler._

  def sampleHiddenGivenVisible(v: INDArray) = {
    val mean = rbm.propUp(v)

    Sample(mean, sample(mean, isBinomialHidden))
  }

  def sampleVisibleGivenHidden(h: INDArray) = {
    val mean = rbm.propDown(h)

    Sample(mean, sample(mean, isBinomialVisible))
  }

  def sampleHiddenVisibleHidden(h: INDArray) = {
    val vh = sampleVisibleGivenHidden(h)
    val hv = sampleHiddenGivenVisible(vh.sample)

    HVHSample(vh, hv)
  }

  private def sample(mean: INDArray, isBinomial: Boolean) =
    binomial(mean, 1, rng)
}