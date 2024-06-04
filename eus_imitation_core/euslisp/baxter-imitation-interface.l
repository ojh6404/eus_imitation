#!/usr/bin/env roseus
(require :baxter-interface "package://baxtereus/baxter-interface.l")
(require :robot-imitation-interface "package://eus_imitation_core/euslisp/lib/robot-imitation-interface.l")

(defclass baxter-imitation-interface
  :super robot-imitation-interface
  :slots ())

(defmethod baxter-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *baxter*)
    (send self :set-val 'robot-name "baxter")
    (send self :set-val 'base-frame-id "base")

   ;; Baxter IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3))

  ;; Baxter
  (:update-current-gripper-pos ()
        (if (send robot :larm)
            (send self :set-arm-val :larm :gripper-pos
                    0.0))
        (if (send robot :rarm)
            (send self :set-arm-val :rarm :gripper-pos
                    0.0)))
  (:angle-vector (&rest args)
    (send* robot-interface :angle-vector-raw args))
  (:inverse-kinematics (arm &rest args)
    (if (eq arm :arms)
      (send* robot :inverse-kinematics-raw (list (send self :get-arm-val :larm :target-coords)
                                                 (send self :get-arm-val :rarm :target-coords))
             :rotation-axis (list t t)
             :move-target (list (send robot :larm :end-coords) (send robot :rarm :end-coords))
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args)
      (send* robot :inverse-kinematics (send self :get-arm-val arm :target-coords)
             :rotation-axis t :move-target (send robot arm :end-coords)
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args))))


(provide :baxter-imitation-interface)
