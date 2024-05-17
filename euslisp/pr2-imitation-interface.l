#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/lib/robot-imitation-interface.l")

(defclass pr2-imitation-interface
  :super robot-imitation-interface
  :slots ())

(defmethod pr2-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *pr2*)
    (send self :set-val 'robot-name "pr2")
    (send self :set-val 'base-frame-id "base_footprint")

   ;; PR2 IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)))

(provide :pr2-imitation-interface)
