#!/usr/bin/env roseus
(require :kinovaeus "package://kinovaeus/kinova-interface.l")
(require :kinova-spacenav-interface "package://eus_imitation/euslisp/kinova-spacenav-interface.l")

(defun init-kinova-imitator (&key (imitation nil) (device :spacenav))
  (kinova-init :type :gen3_robotiq_2f_140)
  (objects (list *kinova*))
  (setq *imitator* (instance kinova-spacenav-interface :init :imitation imitation :device device))
  (send *imitator* :ros-init))
(defun main (&key (imitation nil) (device :spacenav))
  (init-kinova-imitator :imitation imitation :device device)
  (send *imitator* :main-loop :imitation imitation))
