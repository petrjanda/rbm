package nn.trainer

import nn.RBM
import nn.trainer.sampling.GibbsSampler
import org.apache.commons.math3.random.MersenneTwister
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.concurrent.{ExecutionContext, Future}

object ContrastiveDivergence {
  def diff(nn:RBM, v: INDArray, k: Int)(implicit ec:ExecutionContext, rng:MersenneTwister): Future[RBMGradient] = Future {
    val gibbs = new GibbsSampler(nn)
    val inputSample = gibbs.sampleHiddenGivenVisible(v)
    
    val hvhSample = Range(0, k).foldLeft(
      GibbsSampler.HVHSample(
        GibbsSampler.Sample(
          Nd4j.zeros(v.rows, nn.numVisible), 
          Nd4j.zeros(v.rows, nn.numHidden)
        ), inputSample
      )
    ) ( (old, _) => {
      gibbs.sampleHiddenVisibleHidden(old.hvSample)
    })

    calculateDiff(v, hvhSample, inputSample)
  }

  def calculateDiff(input: INDArray, hvhSample: GibbsSampler.HVHSample, inputSample: GibbsSampler.Sample): RBMGradient = {
    val weights = input.transpose().mmul(inputSample.mean).sub(hvhSample.hvMean.transpose().mmul(hvhSample.vhSample))
    val hBias = inputSample.sample.sub(hvhSample.hvMean).mean(0)
    val vBias = input.sub(hvhSample.vhSample).mean(0)

    RBMGradient(weights, vBias, hBias)
  }
}