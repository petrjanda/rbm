package nn.trainer

import nn.RBM
import nn.fn.act.Sigmoid
import nn.trainer.rbm.RBMGradient
import nn.trainer.sampling.GibbsSampler
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object ContrastiveDivergence {
  def diff(nn:RBM, input: INDArray, k: Int)(implicit rng:MersenneTwister) = {
    val gibbs = new GibbsSampler(nn)
    val probHidden = gibbs.sampleHiddenGivenVisible(input)
    
    val hvh = Range(0, k).foldLeft(
      GibbsSampler.HVHSample(
        GibbsSampler.Sample(
          Nd4j.zeros(input.rows, nn.numInputs),
          Nd4j.zeros(input.rows, nn.numOutputs)
        ), probHidden
      )
    ) ( (old, _) => {
      gibbs.sampleHiddenVisibleHidden(old.nhSample)
    })

    val wGradient = input.transpose.mmul(probHidden.sample).sub(hvh.nvSamples.transpose.mmul(hvh.nhMeans))
    val hBiasGradient = probHidden.sample.sub(hvh.nhMeans).mean(0)
    val vBiasGradient = input.sub(hvh.nvSamples).mean(0)

    RBMGradient(wGradient, vBiasGradient, hBiasGradient)
  }
}