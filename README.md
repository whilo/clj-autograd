# clj-autograd

These are some experiments to bring
a
[pytorch](http://pytorch.org/docs/master/index.html)-like
[autograd](http://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/) library
to Clojure. This is not primarily a neural network library, but rather a means
of abstraction to calculate gradients for scientific computing. `clj-autograd`
uses [neanderthal](http://neanderthal.uncomplicate.org/) at the moment, because
we want to be as fast as possible with matrices to allow to build competitive
deep learning and Bayesian inference on top. A core.matrix backend is also
possible, feel free to drop
in [slack](https://clojurians.slack.com/messages/C08PLCRGT/details/)
or [gitter](https://gitter.im/metasoarous/clojure-datascience) and ask me
questions.

Do not consider the API stable yet, there will be quite some changes to a terse 
math notation.


## Usage

To build a logistic regression model and do gradient descent with the automatic
gradient, you can do something like this:

~~~clojure
(let [X (tape (trans (dge 2 3 [5 2 -1 0 5 2]))) ;; toy training data
      Y (tape (dv [1 0 1])) ;; toy labels
      c (tape 1 true) ;; intercept
      b (tape (dv [1 1]) true)] ;; weight vector
    (loop [i 1000]
      (when (pos? i)
        (let [Y* (sigmoid (add (mul X b) (broadcast-like c Y))) ;; calculate estimator
              out (bcrossent Y* Y) ;; use binary cross-entropy loss
              grads ((:backward out) out 1)] ;; calculate gradient
          (when (zero? (mod i 100))
            (prn "Loss:" @(:data out) ", b:"  @(:data b) ", c:" @(:data c)))
          (gd! grads) ;; apply gradient descent (mutates tape values)
          (recur (dec i)))))
    b)
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
