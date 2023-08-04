#!/usr/bin/env roseus
(require :kinova-imitation-interface "package://eus_imitation/euslisp/kinova-imitation-interface.l")


(defun init-kinova-imitator ()
  (kinova-init :type :gen3_robotiq_2f_140)
  (objects (list *kinova*))
  (setq *kinova-imitator* (instance kinova-imitation-interface :init))
  (send *kinova-imitator* :ros-init))
(defun main (&key (arm :rarm) (imitation nil) (rot nil))
  (init-kinova-imitator)
  (send *kinova-imitator* :main-loop :arm arm :imitation imitation :rot rot))
