(ns clj-autograd.core-test
  (:require [clojure.test :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clj-autograd.core :refer :all]))

(def eps 10E-4)

(defn check-vector-gradient
  "Checks a function of a vector numerically. A good description can be found here:
  http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/"
  [a out-fn]
  (let [n (dim @(:data a))
        out (out-fn a)
        scalar? (number? @(:data out))
        grads ((:backward out) out (if scalar? 1.0 (dv (repeat n 1))))]
    (not
     (some #(< eps %)
           (for [i (range n)]
             (let [out-m (out-fn (add a (tape (dv (assoc (vec (repeat n 0)) i eps)))))
                   out-p (out-fn (add a (tape (dv (assoc (vec (repeat n 0)) i (- eps))))))
                   out-delta (if scalar?
                               (- @(:data out-m) @(:data out-p))
                               (sub out-m out-p))]
               (Math/abs
                (- (-> grads :children first :grad (entry i))
                   ;; numerical gradient
                   (/ (if scalar?
                        out-delta
                        (sum @(:data out-delta))) (* 2 eps))))))))))

(deftest broadcast-test
  (testing "Test broadcast."
    (let [a (tape (dv [5 5 5]))
          c (tape 5 true)
          out (broadcast-like c a)
          grads ((:backward out) out (dv [1 1 1]))]
      (is (= @(:data out)
             @(:data a)))
      (is (= (-> grads :children first :grad)
             3.0))
      )))

(deftest forward-test
  (testing "Test normal function values."
    (let [a (tape (dv [1.0 0.0 -3.21]))
          b (tape (dv [-1.3 2.8 -0.1]))
          <eps (fn [a b] (< (Math/abs (sum (axpy @(:data a) (dv (map - b))))) eps))]
      (is (= @(:data (inner a a)) 11.3041))
      (is (= @(:data (inner a b)) -0.9790000000000001))
      (is (<eps (add a b) [-0.30 2.80 -3.31]))
      (is (<eps (sub a b) [2.3 -2.8 -3.11]))
      (is (<eps (emul a b) [-1.3 0.0 0.321]))
      (is (<eps (mul (tape (dge 3 3 [1 0 0
                                     0 2 0
                                     0 0 3])) a)
                [1.0 0.0 -9.63]))
      (is (<eps (sigmoid a) [0.73 0.50 0.04]))
      (is (<eps (log (tape (dv [0.2 0.9 3.21])))
                [-1.6094379124341003 -0.10536051565782628 1.1662709371419244]))
      (is (= @(:data (bcrossent (tape (dv [0.1 0.5 0.9]))
                                (tape (dv [0 0 1]))))
             0.9038682118755978)))))

(deftest mul-test
  (testing "Matrix vector multiplication."
    (let [m (tape (dge 2 2 [2 0 0 2]) true)
          v (tape (dv [2 3]))
          out (mul m v)
          grads ((:backward out) out (dv [1 1]))]
      (is (= (-> grads :children first :grad seq)
             '((2.0 2.0) (3.0 3.0)))))))

(deftest mmul-test
  (testing "Matrix matrix multiplication."
    (let [A (dge 2 3 (repeat (* 2 6) 1))
          B (dge 3 1 (repeat 3 0.1))
          delta (dge 2 1 (repeat 2 1.0))
          out (sigmoid (mmul (tape A true) (tape B true)))
          grads ((:backward out) out delta #_(trans m))
          ]
      (is (= (seq (-> grads :children first :grad))
             '((0.24445831169074586 0.24445831169074586))))
      #_@(:data out))))


(deftest vector-gradients
  (testing "Test functions that take a vector as first argument."
    (let [to-check (tape (dv [1.2 0.0 -5.8]) true)
          to-check-p (tape (dv [0.1 0.9 3.2]) true)
          other (tape (dv (range 3)))]
      (is (check-vector-gradient to-check #(add % other)))
      (is (check-vector-gradient to-check #(sub % other)))
      (is (check-vector-gradient to-check #(emul % other)))
      (is (check-vector-gradient to-check #(sigmoid %)))
      (is (check-vector-gradient to-check #(inner % other)))
      (is (check-vector-gradient to-check-p #(log %)))
      (is (check-vector-gradient to-check-p #(bcrossent % (tape (dv [1 0 1]))))))))


(deftest linear-regression
  (testing "A linear regression model with euclidean loss."
    (let [X (tape (trans (dge 1 2 [1 3])))
          Y (tape (dv [-10 -30]))
          c (tape 0 true)
          b (tape (dv [10]) true)]
      (loop [i 1000]
        (when (pos? i)
          (let [out (sub (add (mul X b) (broadcast-like c Y)) Y)
                out (inner out out)
                grads ((:backward out) out 1)]
            #_(when (zero? (mod i 100))
              (prn "Loss:" @(:data out) ", b:"  @(:data b) ", c:" @(:data c)))
            (gd! grads)
            (recur (dec i)))))
      (is (= (seq @(:data (add (mul X b) (broadcast-like c Y))))
             '(-10.004230810867261 -29.998247540758957))))))

(deftest logistic-regression
  (testing "Train a logistic regression."
    (let [X (tape (trans (dge 2 3 [5 2 -1 0 5 2])))
          Y (tape (dv [1 0 1]))
          c (tape 0 true)
          b (tape (dv [0 0]) true)]
      (loop [i 1000]
        (when (pos? i)
          (let [Y* (sigmoid (add (mul X b) (broadcast-like c Y)))
                out (bcrossent Y* Y)
                grads ((:backward out) out 1)]
            #_(when (zero? (mod i 100))
              (prn "Loss:" @(:data out) ", b:"  @(:data b) @(:data c)))
            (gd! grads)
            (recur (dec i)))))
      (is (= (seq @(:data (sigmoid (add (mul X b) (broadcast-like c Y)))))
             '(0.9998655456159079 0.04813488562019285 0.9998655456159079))))))

