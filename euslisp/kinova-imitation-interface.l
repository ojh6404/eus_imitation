#!/usr/bin/env roseus
(load "package://kinovaeus/kinova-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")

(defclass kinova-imitation-interface
  :super robot-imitation-interface
  :slots (
          ))


(defmethod kinova-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *kinova*) ;; TODO

   ;; KINOVA control settings
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




  ;; arms settings
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  ;; (send self :get-arm-val :larm :target-end-pos)
                  ;; (send self :get-arm-val :larm :target-end-rpy)
                  ;; (float-vector (send self :get-arm-val :larm :grasp))
                  (v- (send self :get-arm-val :rarm :target-end-pos) (send self :get-arm-val :rarm :current-end-pos))
                  ;; (v- (coerce (send self :get-arm-val :rarm :target-end-rpy) float-vector) (coerce (send self :get-arm-val :rarm :current-end-rpy) float-vector))
                  ;; (send self :get-arm-val :rarm :target-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :grasp))
                  ))) ;; (3 + 3 + 1) * 2 = 14

  (:update-robot-state ()
        (send self :set-val 'robot-state ;; you can define agent's state, proprioception
              (concatenate float-vector ;; concatenate all vectors
                  ;; (send self :get-arm-val :larm :current-end-pos)
                  ;; (send self :get-arm-val :larm :current-end-rpy)
                  ;; (float-vector (send self :get-arm-val :larm :gripper-pos))
                  (send self :get-arm-val :rarm :current-end-pos)
                  ;; (send self :get-arm-val :rarm :current-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :gripper-pos))
                  ))) ;; (3 + 3 + 1) * 2 = 14

  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let (action rarm-target-end-effector-coords rarm-target-end-effector-pos rarm-gripper-command)
        (setq action (send self :get-val 'robot-action))
        ;; parsing action
        (setq rarm-target-end-effector-pos (subseq action 0 3))
        (setq rarm-target-end-effector-rpy (float-vector pi/2 pi/2 -pi/2))
        (setq rarm-gripper-command (elt action 3))
        ;; (setq target-end-effector-coords (make-coords :pos larm-target-end-effector-pos :rpy larm-target-end-effector-rpy))
        ;; (setq target-end-effector-coords (make-coords :pos (v+ rarm-target-end-effector-pos (send self :get-arm-val :rarm :current-end-pos)) :rpy (v+ (coerce rarm-target-end-effector-rpy float-vector) (coerce (send self :get-arm-val :rarm :current-end-rpy) float-vector))))
        (setq target-end-effector-coords (make-coords :pos (v+ rarm-target-end-effector-pos (send self :get-arm-val :rarm :current-end-pos)) :rpy rarm-target-end-effector-rpy))
        ;; execute action
        (send self :inverse-kinematics :rarm target-end-effector-coords)
        (send self :angle-command-vector :rarm :torso nil)
        ;; (send self :move-gripper :rarm larm-gripper-pos :delta nil :wait nil)
        (if (> rarm-gripper-command 0.01)
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil))
        ))



  )


(provide :kinova-imitation-interface)
