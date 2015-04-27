package nn.trainer

import nn.RBM
import nn.trainer.sampling.GibbsSampler
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object ContrastiveDivergence {
  def diff(nn:RBM, input: INDArray, k: Int)(implicit rng:MersenneTwister) = {
    val gibbs = new GibbsSampler(nn)
    val inputSample = gibbs.sampleHiddenGivenVisible(input)
    
    val hvhSample = Range(0, k).foldLeft(
      GibbsSampler.HVHSample(
        GibbsSampler.Sample(
          Nd4j.zeros(input.rows, nn.numVisible),
          Nd4j.zeros(input.rows, nn.numHidden)
        ), inputSample
      )
    ) ( (old, _) => {
      gibbs.sampleHiddenVisibleHidden(old.hvSample)
    })

    val weights = input.transpose().mmul(inputSample.mean).sub(hvhSample.hvMean.transpose().mmul(hvhSample.vhSample))
    val hBias = inputSample.sample.sub(hvhSample.hvMean).mean(0)
    val vBias = input.sub(hvhSample.vhSample).mean(0)

    RBMGradient(weights, vBias, hBias)
  }
}