#!/usr/bin/env roseus
(load "package://kinovaeus/kinova-interface.l")
;; (require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-spacenav-interface.l")

(defclass kinova-imitation-interface
  :super robot-spacenav-interface
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

   (send self :set-val 'robot-joint-states-topic-name "/arm_gen3/joint_states")
   )

  ;; (:test-cb (msg)
  ;;   ;; (send self :set-val 'joint-pos (send msg :position))
  ;;   ;; (send self :set-val 'joint-vel (send msg :velocity))
  ;;   ;; (send self :set-val 'joint-efforts (send msg :effort))

  ;;   ;; (send self :set-arm-val :rarm :gripper-pos (elt (send self :get-val 'joint-positions) 7)))
  ;;   (send self :set-arm-val :rarm :gripper-pos (elt (send msg :position) 7)))

  ;; arms settings
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-linear-velocity)
                  (send self :get-arm-val :rarm :target-angular-velocity)
                  (float-vector (send self :get-arm-val :rarm :grasp))))) ;; (3 + 3 + 1) * 2 = 14

  (:update-robot-state ()
        (send self :set-val 'robot-state ;; you can define agent's state, proprioception
              (concatenate float-vector ;; concatenate all vectors
                  (send self :get-arm-val :rarm :current-end-pos)
                  (send self :get-arm-val :rarm :current-end-rpy)
                  ;; (float-vector (send self :get-arm-val :rarm :gripper-pos)) ;;TODO
                  ))) ;; (3 + 3 + 1) * 2 = 14

  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let (action target-linear-velocity target-angular-velocity gripper-command)
        (setq action (send self :get-val 'robot-action))
        ;; parsing action
        (setq target-linear-velocity (subseq action 0 3))
        (setq target-angular-velocity (subseq action 3 6))
        (setq gripper-command (elt action 6))
        (send self :cartesian-velocity target-linear-velocity target-angular-velocity)
        ;; (if (> gripper-command 0.01) ;; TODO
        ;;     (send self :start-grasp :rarm :wait nil)
        ;;     (send self :stop-grasp :rarm :wait nil))

        ))


  (:publish-robot-state () ;; TODO
    (setq robot-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
    (send robot-state-msg :data (send self :get-val 'robot-state))
    (send robot-state-msg :header :stamp (ros::time-now))
    (ros::publish robot-state-topic-name robot-state-msg)
    (if (send robot :larm)
        (progn
            (setq robot-larm-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
            (send robot-larm-state-msg :data (concatenate float-vector (send self :get-arm-val :larm :current-end-pos) (send self :get-arm-val :larm :current-end-rpy) (float-vector (send self :get-arm-val :larm :gripper-pos)) (send self :get-arm-val :larm :joint-state)))
            (send robot-larm-state-msg :header :stamp (ros::time-now))
            (ros::publish robot-larm-state-topic-name robot-larm-state-msg)))
    (if (send robot :rarm)
        (progn
            (setq robot-rarm-state-msg (instance eus_imitation::Float32MultiArrayStamped :init))
            (send robot-rarm-state-msg :data (concatenate float-vector (send self :get-arm-val :rarm :current-end-pos) (send self :get-arm-val :rarm :current-end-rpy) (send self :get-arm-val :rarm :joint-state)))
            (send robot-rarm-state-msg :header :stamp (ros::time-now))
            (ros::publish robot-rarm-state-topic-name robot-rarm-state-msg))))

  (:update-current-gripper-pos () ;; TODO
      (if (send robot :larm)
        (send self :set-arm-val :larm :gripper-pos
              (send robot-interface :gripper :larm :position)))
      (if (send robot :rarm)
        (send self :set-arm-val :rarm :gripper-pos
              (send robot-interface :gripper :rarm :position))))
  (:update-robot () ;; TODO
    (send self :update-robot-model)
    (send self :update-current-end-coords)
    ;; (send self :update-current-gripper-pos) ;; TODO
    (send self :update-current-joint-state)
    (send self :update-robot-state)) ;; update robot state for IL or RL, defined by user


  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp :wait wait))

  (:cartesian-velocity (linear rotation)
     (send robot-interface :cartesian-velocity linear rotation))

  (:move-robot () ;; move real robot
      (if (and data-collection control-arm) ;; TODO loop-enable : when one of arm is enabled
        (progn ;; when data collection is enabled and arm control is enabled
          (send self :get-target-cartesian-velocity control-arm)
          (send self :update-robot-action)
          (send self :publish-robot-action)

          (print (send *kinova* :angle-vector))

          ;; (send self :cartesian-velocity (send self :get-arm-val control-arm :target-linear-velocity) #f(0.0 0.0 0.0))
          (send self :cartesian-velocity (send self :get-arm-val control-arm :target-linear-velocity)
                                         (send self :get-arm-val control-arm :target-angular-velocity))

          ;; (if (> (send self :get-arm-val :rarm :grasp) 0.5) ;; TODO
          ;;     (send self :start-grasp :rarm :wait nil)
          ;;     (send self :stop-grasp :rarm :wait nil))
          )
        (if (and (not data-collection) (send self :get-val 'robot-action)) ;; when imitation, parsing action from topic recieved and execute
          (send self :parse-robot-action))))

  )

(provide :kinova-imitation-interface)
