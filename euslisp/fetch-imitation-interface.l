#!/usr/bin/env roseus
(require :fetch-interface "package://fetcheus/fetch-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/lib/robot-imitation-interface.l")

(defclass fetch-imitation-interface
  :super robot-imitation-interface
  :slots ())

(defmethod fetch-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *fetch*)
    (send self :set-val 'robot-name "fetch")
    (send self :set-val 'base-frame-id "base_link")

   ;; fetch IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3))

  ;; NOTE fetch has only one arm
  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp :wait wait))
  (:update-current-gripper-pos ()
     (if (send robot :rarm)
         (send self :set-arm-val :rarm :gripper-pos
               (if (send robot-interface :gripper :position)
                   (send robot-interface :gripper :position)
                   0.0))))
  (:inverse-kinematics (arm target-coords &rest args)
    (send* robot arm :inverse-kinematics target-coords
            :rotation-axis t :move-target (send robot arm :end-coords)
            :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
            :debug-view nil args))

  ;; angle-vector-raw
  (:angle-vector (&rest args)
    (send* robot-interface :angle-vector-raw args))
  (:angle-command-vector (arm &key (torso nil))
      (send self :angle-vector (send robot :angle-vector) av-tm)
      (if torso
        (send self :angle-vector (send robot :angle-vector)
              torso-av-tm :torso-controller 0
              :min-time torso-min-time :scale torso-av-scale))))

(provide :fetch-imitation-interface)
