#!/usr/bin/env roseus
(require :kinova-imitation-interface "package://eus_imitation/euslisp/kinova-imitation-interface.l")

(defun init-kinova-imitator (&key (imitation nil))
  (kinova-init :type :gen3_robotiq_2f_140)
  (objects (list *kinova*))
  (setq *kinova-imitator* (instance kinova-imitation-interface :init :imitation imitation))
  (send *kinova-imitator* :ros-init))
(defun main (&key (imitation nil) )
  (init-kinova-imitator :imitation imitation)
  (send *kinova-imitator* :main-loop :imitation imitation))
