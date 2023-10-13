#!/usr/bin/env roseus
(require :kinovaeus "package://kinovaeus/kinova-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")

(defclass kinova-imitation-interface
  :super robot-imitation-interface
  :slots ())

(defmethod kinova-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *kinova*)
    (send self :set-val 'robot-name "kinova")

   ;; KINOVA IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'scale 1.5))
  (:ros-init ()
    (send-super :ros-init)
    (ros::subscribe "/eus_imitation/gripper_state"
                    eus_imitation::Float32MultiArrayStamped #'send self :gripper-pos-cb) ;; TODO currently only rarm
    t)
  (:gripper-pos-cb (msg)
            (send self :set-arm-val :rarm :gripper-pos (elt (send msg :data) 0)))
  ;; arms settings
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-linear-velocity)
                  (float-vector 0.0 0.0 0.0)
                  ;; (send self :get-arm-val :rarm :target-angular-velocity)
                  (float-vector (send self :get-arm-val :rarm :grasp))))) ;; (3 + 3 + 1) * 2 = 14
  (:update-robot-state ()
        (send self :set-val 'robot-state ;; you can define agent's state, proprioception
              (concatenate float-vector ;; concatenate all vectors
                  (send self :get-arm-val :rarm :current-end-pos)
                  (send self :get-arm-val :rarm :current-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :gripper-pos))))) ;; (3 + 3 + 1) * 2 = 14
  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let (action target-linear-velocity target-angular-velocity gripper-command)
        (setq action (send self :get-val 'robot-action))
        ;; parsing action
        (setq target-linear-velocity (subseq action 0 3))
        ;; (setq target-angular-velocity (subseq action 3 6))
        (setq target-angular-velocity (float-vector 0.0 0.0 0.0))
        (setq gripper-command (elt action 6))
        (send self :cartesian-velocity target-linear-velocity target-angular-velocity)
        ;; (if (> gripper-command 0.5)
        ;;     (send self :start-grasp :rarm :wait nil)
        ;;     (send self :stop-grasp :rarm :wait nil))
        ))
  (:update-current-gripper-pos ()) ;; NOTE dummy cause gripper pos cb
  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp :wait wait))
  (:cartesian-velocity (linear rotation)
     (send robot-interface :cartesian-velocity linear rotation))
  (:move-robot () ;; move real robot
      (if (and data-collection control-arm)
        (progn ;; when data collection is enabled and arm control is enabled
          (send self :get-target-cartesian-velocity control-arm)
          (send self :update-robot-action)
          (send self :publish-robot-action)
          ;; (send self :cartesian-velocity (send self :get-arm-val control-arm :target-linear-velocity)
          ;;                                (send self :get-arm-val control-arm :target-angular-velocity))
          (send self :cartesian-velocity (send self :get-arm-val control-arm :target-linear-velocity) (float-vector 0.0 0.0 0.0))

          ;; (if (> (send self :get-arm-val :rarm :grasp) 0.5)
          ;;     (send self :start-grasp :rarm :wait nil)
          ;;     (send self :stop-grasp :rarm :wait nil))
          )
        (if (and (not data-collection) (send self :get-val 'robot-action)) ;; when imitation, parsing action from topic recieved and execute
          (send self :parse-robot-action)))))

  ;; ;; NOTE user can define action and state
  ;; (:update-robot-action () ;; when data collection
  ;;       (send self :set-val 'robot-action
  ;;             (concatenate float-vector
  ;;                 (send self :get-arm-val :rarm :target-end-pos)
  ;;                 (send self :get-arm-val :rarm :target-end-rpy)
  ;;                 (float-vector (send self :get-arm-val :rarm :grasp))
  ;;                 (float-vector (send self :get-arm-val :rarm :grasp)))))
  ;; (:update-robot-state ()
  ;;       (send self :set-val 'robot-state
  ;;             (concatenate float-vector
  ;;                 (send self :get-arm-val :rarm :current-end-pos)
  ;;                 (send self :get-arm-val :rarm :current-end-rpy)
  ;;                 (float-vector (send self :get-arm-val :rarm :gripper-pos)))))
  ;; (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
  ;;     (let* ((action (send self :get-val 'robot-action))
  ;;            (rarm-target-end-effector-pos (subseq action 0 3))
  ;;            (rarm-target-end-effector-rpy (subseq action 3 6))
  ;;            (rarm-gripper-command (elt action 6))
  ;;            (rarm-target-end-effector-coords (make-coords :pos rarm-target-end-effector-pos :rpy rarm-target-end-effector-rpy)))
  ;;       ;; execute action
  ;;       (send self :inverse-kinematics :rarm rarm-target-end-effector-coords)
  ;;       (send self :angle-command-vector :rarm :torso nil)
  ;;       (if (> rarm-gripper-command (send self :get-val 'gripper-threshold))
  ;;           (send self :start-grasp :rarm :wait nil)
  ;;           (send self :stop-grasp :rarm :wait nil)))))

(provide :kinova-imitation-interface)
