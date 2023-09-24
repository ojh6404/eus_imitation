#!/usr/bin/env roseus
(require :fetch-interface "package://fetcheus/fetch-interface.l")
(require :robot-vive-interface "package://eus_imitation/euslisp/robot-vive-interface.l")

(defclass fetch-imitation-interface
  :super robot-vive-interface
  :slots ())


(defmethod fetch-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *fetch*)

   ;; fetch IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'scale 1.5))


  (:update-current-gripper-pos ()
        (if (send robot :larm)
            (send self :set-arm-val :larm :gripper-pos
                    0))
        (if (send robot :rarm)
            (send self :set-arm-val :rarm :gripper-pos
                    0)))

  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp :wait wait))

  (:move-robot () ;; move real robot
      (if (and data-collection control-arm) ;; TODO loop-enable : when one of arm is enabled
        (progn ;; when data collection is enabled and arm control is enabled
          (if (eq control-arm :arms)
                (progn ;; when both arms are enabled
                  (send self :get-target-coords :larm :pos-scale scale)
                  (send self :get-target-coords :rarm :pos-scale scale)
                  (send self :inverse-kinematics :arms (list (send self :get-arm-val :larm :target-coords) (send self :get-arm-val :rarm :target-coords)))
                  )
                (progn ;; when one arm is enabled
                  (send self :get-target-coords control-arm :pos-scale scale)
                  ;; (setq target-end-effector-coords (make-coords :pos (send self :get-arm-val control-arm :target-end-pos) :rpy (float-vector pi/2 pi/2 -pi/2)))
                  (setq target-end-effector-coords (make-coords :pos (send self :get-arm-val control-arm :target-end-pos) :rpy (send self :get-arm-val control-arm :target-end-rpy)))
                  (send self :inverse-kinematics control-arm target-end-effector-coords)
                  (send self :update-robot-action)
                  (send self :publish-robot-action)
                  ))
          ;; (send self :angle-command-vector control-arm :torso nil)
            (send *ri* :angle-vector-raw (send robot :angle-vector)
                        av-tm)
          (if (> (send self :get-arm-val :larm :grasp) 0.5)
              (send self :start-grasp :larm :wait nil)
              (send self :stop-grasp :larm :wait nil))
          (if (> (send self :get-arm-val :rarm :grasp) 0.5)
              (send self :start-grasp :rarm :wait nil)
              (send self :stop-grasp :rarm :wait nil))
          )
        (if (and (not data-collection) (send self :get-val 'robot-action)) ;; when imitation, parsing action from topic recieved and execute
          (send self :parse-robot-action)))
               )

  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-end-pos)
                  (send self :get-arm-val :rarm :target-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :grasp))
                  (float-vector (send self :get-arm-val :rarm :grasp)))))

  (:update-robot-state ()
        (send self :set-val 'robot-state ;; you can define agent's state, proprioception
              (concatenate float-vector ;; concatenate all vectors
                  (send self :get-arm-val :rarm :current-end-pos)
                  (send self :get-arm-val :rarm :current-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :gripper-pos)))))

  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let (action rarm-target-end-effector-coords rarm-target-end-effector-pos rarm-gripper-command)
        (setq action (send self :get-val 'robot-action))
        ;; parsing action
        (setq rarm-target-end-effector-pos (subseq action 0 3))
        (setq rarm-target-end-effector-pos (subseq action 3 6))
        (setq rarm-gripper-command (elt action 6))
        (setq target-end-effector-coords (make-coords :pos rarm-target-end-effector-pos :rpy rarm-target-end-effector-rpy))
        ;; execute action
        (send self :inverse-kinematics :rarm target-end-effector-coords)
        (send self :angle-command-vector :rarm :torso nil)
        (if (> rarm-gripper-command 0.01)
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil))
        ))
  )

(provide :fetch-imitation-interface)
