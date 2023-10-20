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

   ;; PR2 IK control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    ;; (send self :set-val 'gripper-pos-scale 90.0))
    (send self :set-val 'gripper-pos-scale 1.0))

  ;; NOTE user can define action and state
  (:move-robot () ;; move real robot
      (if (and data-collection control-arm)
        (progn ;; when data collection is enabled and arm control is enabled
          (if (eq control-arm :arms)
                (progn ;; when both arms are enabled
                  (send self :get-target-coords :larm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :get-target-coords :rarm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :inverse-kinematics :arms (list (send self :get-arm-val :larm :target-coords) (send self :get-arm-val :rarm :target-coords))))
                (progn ;; when one arm is enabled
                  (send self :get-target-coords control-arm :pos-scale pos-scale :rot-scale rot-scale)
                  (send self :inverse-kinematics control-arm (send self :get-arm-val control-arm :target-coords))
                  (send self :update-robot-action)
                  (send self :publish-robot-action)))
          (send self :angle-command-vector control-arm :torso nil)
          (if (> (send self :get-arm-val :larm :grasp) gripper-threshold)
              (send self :start-grasp :larm :wait nil)
              (send self :stop-grasp :larm :wait nil))
          (if (> (send self :get-arm-val :rarm :grasp) gripper-threshold)
              (send self :start-grasp :rarm :wait nil)
              (send self :stop-grasp :rarm :wait nil)))
        (if (and (not data-collection) robot-action) ;; when imitation, parsing action from topic recieved and execute
          (send self :parse-robot-action))))
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (if object-centric
                (concatenate float-vector
                    (v- (send self :get-arm-val :rarm :target-end-pos) (send self :get-val 'ref-pos))
                    (send self :get-arm-val :rarm :target-end-rpy) ;; TODO
                    (float-vector (send self :get-arm-val :rarm :grasp)))
                (concatenate float-vector
                    (send self :get-arm-val :rarm :target-end-pos)
                    (send self :get-arm-val :rarm :target-end-rpy)
                    (float-vector (send self :get-arm-val :rarm :grasp))))))
  (:update-robot-state ()
        (send self :set-val 'robot-state
              (if object-centric
                (concatenate float-vector
                    (v- (send self :get-arm-val :rarm :current-end-pos) (send self :get-val 'ref-pos))
                    (send self :get-arm-val :rarm :current-end-rpy)
                    (float-vector (send self :get-arm-val :rarm :gripper-pos)))
                (concatenate float-vector
                    (send self :get-arm-val :rarm :current-end-pos)
                    (send self :get-arm-val :rarm :current-end-rpy)
                    (float-vector (send self :get-arm-val :rarm :gripper-pos))))))
  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let* ((action (send self :get-val 'robot-action))
             (rarm-target-end-effector-pos (if object-centric
                                               (v+ (coerce (subseq action 0 3) float-vector) (send self :get-val 'ref-pos))
                                               (subseq action 0 3)))
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
