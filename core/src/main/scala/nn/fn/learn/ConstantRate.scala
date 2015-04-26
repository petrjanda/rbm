package nn.fn.learn

class ConstantRate(rate: Double) extends LearningFunction {
  def apply(iteration: Int): Double = rate
}

object ConstantRate {
  def apply(rate: Double): ConstantRate = new ConstantRate(rate)
}