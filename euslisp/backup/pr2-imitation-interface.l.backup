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

          ;; (setf robot-action-real (concatenate float-vector (send (send (send self :get-arm-val arm :current-end-coords) :copy-worldcoords) :worldpos ) (float-vector 0.0))) ;; 3 + 1
          ;; (ros::subscribe "/eus_imitation/policy_action" eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (setq robot-action-real (send msg :data))) 1)))

  (:execute-action () ;; define own action execution
      (let (action target-end-effector-coords target-rpy gripper-command)
        (setq action (send self :get-val 'robot-action))
        (setq target-rpy (float-vector pi/2 pi/2 -pi/2))
        (setq target-end-effector-coords (make-coords :pos (subseq action 0 3) :rpy target-rpy))
        (setq gripper-command (elt action 3))
        (send self :inverse-kinematics :rarm target-end-effector-coords)
        (send self :angle-command-vector :rarm :torso nil)
        (if (> gripper-command 0.5)
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil))))

  (:update-robot-action () ;; when data collection
      (let (target-end-effector-pos gripper-command)
        (setq target-end-effector-coords (send (send self :get-target-coords :rarm :pos-scale 1.0) :copy-worldcoords))
        (setq target-end-effector-pos (send target-end-effector-coords :worldpos))
        (setq gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
        (send self :set-val 'robot-action (concatenate float-vector target-end-effector-pos gripper-command))))

  (:update-robot-state ()
      (let (end-effector-pos gripper-pos)
        (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
        (setq gripper-command (float-vector (send self :get-arm-val :rarm :grasp)))
        (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-command))))

  ;; (:update-robot-state ()
  ;;     (let (end-effector-pos gripper-pos)
  ;;       (setq end-effector-pos (send (send (send self :get-arm-val :rarm :current-end-coords) :copy-worldcoords) :worldpos))
  ;;       (setq gripper-pos (float-vector (send self :get-arm-val :rarm :gripper-pos)))
  ;;       (send self :set-val 'robot-state (concatenate float-vector end-effector-pos gripper-pos))))

  )


(provide :pr2-imitation-interface)
