package nn.conf

import nn.fn.act.ActivationFunction
import nn.fn.loss.LossFunction

case class NNConf(activation:ActivationFunction,
                  loss:LossFunction)
