package nn.fn.learn

trait LearningFunction {
  def apply(iteration: Int): Double
}
