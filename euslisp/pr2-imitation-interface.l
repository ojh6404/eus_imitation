#!/usr/bin/env roseus
(require :pr2-interface "package://pr2eus/pr2-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")

(defclass pr2-imitation-interface
  :super robot-imitation-interface
  :slots (
          ))


(defmethod pr2-imitation-interface
  (:init (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *pr2*) ;; TODO

   ;; PR2 control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'scale 1.5)


   )
  (:ros-init ()
    (send-super :ros-init)
    (send self :set-arm-val :larm :gripper-status-topic-name
            "/l_gripper_controller/gripper_action/status")
    (send self :set-arm-val :rarm :gripper-status-topic-name
            "/r_gripper_controller/gripper_action/status")

    )
  )


(provide :pr2-imitation-interface)
