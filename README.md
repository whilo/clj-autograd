# clj-autograd

These are some experiments to bring a [pytorch](http://pytorch.org)-like
autograd
[1](https://arxiv.org/abs/1502.05767) [2](http://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/) library
to Clojure. This is not primarily a neural network library, but rather a means
of abstraction to calculate gradients for scientific computing. `clj-autograd`
is built with [denisovan](https://github.com/whilo/denisovan),
a [core.matrix](https://github.com/mikera/core.matrix) backend
for [neanderthal](http://neanderthal.uncomplicate.org/) at the moment, because
we want to be as fast as possible with linear algebra and build competitive deep
learning and Bayesian inference on top. We hope to keep the general autograd
machinery portable along core.matrix backends, but will explore optionally
inlined neanderthal operations, where they improve performance.

Feel free to drop
in [slack](https://clojurians.slack.com/messages/C08PLCRGT/details/)
or [gitter](https://gitter.im/metasoarous/clojure-datascience) and ask me
questions.

Do not consider the API stable yet, there will be quite some changes to a terse 
math notation.


## Usage

To build a logistic regression model and do gradient descent with the automatic
gradient, you can do something like this:

~~~clojure
(let [X (tape (m/matrix [[5 2] [-1 0] [5 2]]))
          Y (tape (m/matrix [1 0 1]))
          c (tape 0 true)
          b (tape (m/matrix [0 0]) true)]
      (loop [i 1000]
        (when (pos? i)
          (let [Y* (sigmoid (add (mul X b) (broadcast-like c Y)))
                out (bcrossent Y* Y)
                grads ((:backward out) out 1)]
            (when (zero? (mod i 100))
              (prn "Loss:" @(:data out) ", b:"  @(:data b) ", c:" @(:data c)))
            (gd! grads)
            (recur (dec i)))))
      (is (= (seq @(:data (sigmoid (add (mul X b) (broadcast-like c Y)))))
             '(0.9998655456159079 0.04813488562019285 0.9998655456159079))))

~~~

Look also at the tests for more examples.

## TODO

- polymorphic product for scalar, vector, matrix, tensor
- identity of Variable? yes, as atom for inplace ops
- trace shapes
- 1. demo
  + linear regression DONE
  + gradient check DONE
- basic operations to classify MNIST:
- loss: sub, pow DONE
- forward with (mini)batches:
  + mmul DONE
- activations:
- sigmoid DONE
- requires_grad DONE
- optimizer DONE
- lazyness?
- figure out how to do in-place ops?



## License

Copyright © 2017 Christian Weilbach

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
