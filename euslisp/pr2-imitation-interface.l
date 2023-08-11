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


  ;; (:execute-action () ;; define own action execution
  ;;     (let (action target-end-effector-coords target-end-effector-pos target-end-effector-rpy gripper-command)
  ;;       (setq action (send self :get-val 'robot-action))
  ;;       (setq target-end-effector-pos (subseq action 0 3))
  ;;       (setq target-end-effector-rpy (subseq action 3 6))
  ;;       (setq target-end-effector-coords (make-coords :pos target-end-effector-pos :rpy target-end-effector-rpy))
  ;;       (setq gripper-command (elt action 6))
  ;;       (send self :inverse-kinematics :rarm target-end-effector-coords)
  ;;       (send self :angle-command-vector :rarm :torso nil)
  ;;       (if (> gripper-command 0.5)
  ;;           (send self :start-grasp :rarm :wait nil)
  ;;           (send self :stop-grasp :rarm :wait nil))))

  ;; (:update-robot-action () ;; when data collection
  ;;     (let (target-end-effector-pos target-end-effector-rpy gripper-command)
  ;;       (setq target-end-effector-coords (send (send self :get-target-coords :rarm :pos-scale 1.0) :copy-worldcoords))
  ;;       (setq target-end-effector-pos (send target-end-effector-coords :worldpos))
  ;;       (setq target-end-effector-rpy (elt (rpy-angle (send target-end-effector-coords :worldrot)) 0))
  ;;       (setq gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
  ;;       (send self :set-val 'robot-action (concatenate float-vector target-end-effector-pos target-end-effector-rpy gripper-command))))

  ;; for two arms
  (:execute-action () ;; define own action execution
      (let (action larm-target-end-effector-coords rarm-target-end-effector-coords
            larm-target-end-effector-pos rarm-target-end-effector-pos
            larm-target-end-effector-rpy rarm-target-end-effector-rpy
            larm-gripper-command rarm-gripper-command)
        (setq action (send self :get-val 'robot-action))
        (setq larm-target-end-effector-pos (subseq action 0 3))
        (setq larm-target-end-effector-rpy (subseq action 3 6))
        (setq larm-target-end-effector-coords (make-coords :pos larm-target-end-effector-pos :rpy larm-target-end-effector-rpy))
        (setq rarm-target-end-effector-pos (subseq action 6 9))
        (setq rarm-target-end-effector-rpy (subseq action 9 12))
        (setq rarm-target-end-effector-coords (make-coords :pos rarm-target-end-effector-pos :rpy rarm-target-end-effector-rpy))
        (setq larm-gripper-command (elt action 12))
        (setq rarm-gripper-command (elt action 13))
        (send self :inverse-kinematics :arms (list larm-target-end-effector-coords rarm-target-end-effector-coords))
        (send self :angle-command-vector :arms :torso nil)
        (if (> larm-gripper-command 0.5)
            (send self :start-grasp :larm :wait nil)
            (send self :stop-grasp :larm :wait nil))
        (if (> rarm-gripper-command 0.5)
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil))))

  (:update-robot-action () ;; when data collection
      (let (larm-target-end-effector-coords rarm-target-end-effector-coords
            larm-target-end-effector-pos rarm-target-end-effector-pos
            larm-target-end-effector-rpy rarm-target-end-effector-rpy
            larm-gripper-command rarm-gripper-command)
        (setq larm-target-end-effector-coords (send (send self :get-target-coords :larm :pos-scale 1.0) :copy-worldcoords))
        (setq larm-target-end-effector-pos (send larm-target-end-effector-coords :worldpos))
        (setq larm-target-end-effector-rpy (elt (rpy-angle (send larm-target-end-effector-coords :worldrot)) 0))
        (setq rarm-target-end-effector-coords (send (send self :get-target-coords :rarm :pos-scale 1.0) :copy-worldcoords))
        (setq rarm-target-end-effector-pos (send rarm-target-end-effector-coords :worldpos))
        (setq rarm-target-end-effector-rpy (elt (rpy-angle (send rarm-target-end-effector-coords :worldrot)) 0))
        (setq larm-gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
        (setq larm-gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
        (send self :set-val 'robot-action (concatenate float-vector
                                                       larm-target-end-effector-pos larm-target-end-effector-rpy
                                                       rarm-target-end-effector-pos rarm-target-end-effector-rpy
                                                       larm-gripper-command rarm-gripper-command))))



  (:update-robot-state ()
      (let (end-effector-pos gripper-pos)
        (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
        (setq gripper-pos (float-vector (send self :get-arm-val :rarm :gripper-pos)))
        (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-pos))))

  ;; this is for fixed rotation
  ;; (:execute-action () ;; define own action execution
  ;;     (let (action target-end-effector-coords target-rpy gripper-command
  ;;       (setq action (send self :get-val 'robot-action))
  ;;       (setq target-rpy (float-vector pi/2 pi/2 -pi/2))
  ;;       (setq target-end-effector-coords (make-coords :pos (subseq action 0 3) :rpy target-rpy))
  ;;       (setq gripper-command (elt action 3))
  ;;       (send self :inverse-kinematics :rarm target-end-effector-coords)
  ;;       (send self :angle-command-vector :rarm :torso nil)
  ;;       (if (> gripper-command 0.5)
  ;;           (send self :start-grasp :rarm :wait nil)
  ;;           (send self :stop-grasp :rarm :wait nil))))

  ;; (:update-robot-action () ;; when data collection
  ;;     (let (target-end-effector-pos gripper-command)
  ;;       (setq target-end-effector-coords (send (send self :get-target-coords :rarm :pos-scale 1.0) :copy-worldcoords))
  ;;       (setq target-end-effector-pos (send target-end-effector-coords :worldpos))
  ;;       (setq gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
  ;;       (send self :set-val 'robot-action (concatenate float-vector target-end-effector-pos gripper-command))))

  ;; (:update-robot-state ()
  ;;     (let (end-effector-pos gripper-pos)
  ;;       (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
  ;;       (setq gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
  ;;       (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-command))))

  ;; for gripper pos
  ;; (:update-robot-state ()
  ;;     (let (end-effector-pos gripper-pos)
  ;;       (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
  ;;       (setq gripper-pos (float-vector (send self :get-arm-val :rarm :gripper-pos)))
  ;;       (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-pos))))

  )


(provide :pr2-imitation-interface)
