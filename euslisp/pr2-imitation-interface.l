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
    (send self :set-val 'min-time 0.3))

  ;; NOTE user can define action and state
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-end-pos)
                  (send self :get-arm-val :rarm :target-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :grasp)))))
  (:update-robot-state ()
        (send self :set-val 'robot-state
              (concatenate float-vector
                  (send self :get-arm-val :rarm :current-end-pos)
                  (send self :get-arm-val :rarm :current-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :gripper-pos)))))
  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let* ((action (send self :get-val 'robot-action))
             (rarm-target-end-effector-pos (subseq action 0 3))
             (rarm-target-end-effector-rpy (subseq action 3 6))
             (rarm-gripper-command (elt action 6))
             (rarm-target-end-effector-coords (make-coords :pos rarm-target-end-effector-pos :rpy rarm-target-end-effector-rpy)))
        ;; execute action
        (send self :inverse-kinematics :rarm rarm-target-end-effector-coords)
        (send self :angle-command-vector :rarm :torso nil)
        (if (> rarm-gripper-command gripper-threshold)
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil)))))

(provide :pr2-imitation-interface)
