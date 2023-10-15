#!/usr/bin/env roseus
(require :baxter-interface "package://baxtereus/baxter-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")

(defclass baxter-imitation-interface
  :super robot-imitation-interface
  :slots ())

(defmethod baxter-imitation-interface
  (:init  (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *baxter*)
    (send self :set-val 'robot-name "baxter")

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
  (:inverse-kinematics (arm target-coords &rest args)
    (if (eq arm :arms)
      (send* robot :inverse-kinematics-raw target-coords :rotation-axis (list t t)
             :move-target (list (send robot :larm :end-coords) (send robot :rarm :end-coords))
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args)
      (send* robot :inverse-kinematics target-coords
             :rotation-axis t :move-target (send robot arm :end-coords)
             :avoid-collision-distance 5 :revert-if-fail nil :stop ik-stop-step
             :debug-view nil args)))

  ;; NOTE user can define action and state
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-end-pos)
                  (send self :get-arm-val :rarm :target-end-rpy)
                  (float-vector (send self :get-arm-val :rarm :grasp))
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
        (if (> rarm-gripper-command (send self :get-val 'gripper-threshold))
            (send self :start-grasp :rarm :wait nil)
            (send self :stop-grasp :rarm :wait nil)))))

(provide :baxter-imitation-interface)
