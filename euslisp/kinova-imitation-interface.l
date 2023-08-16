#!/usr/bin/env roseus
;; (require :pr2-interface "package://pr2eus/pr2-interface.l")
;; (require :kinova-interface "package://kinovaeus/kinova-interface.l")
(require :robot-imitation-interface "package://eus_imitation/euslisp/robot-imitation-interface.l")
(load "package://kinovaeus/kinova-interface.l")

(defclass kinova-imitation-interface
  :super robot-imitation-interface
  :slots (
          ))


(defmethod kinova-imitation-interface
  (:init (&rest args)
    (send-super* :init args)
    (send self :set-val 'robot *kinova*) ;; TODO

   ;; Kinova control settings
    (send self :set-val 'ik-stop-step 10)
    (send self :set-val 'av-tm 800)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'scale 0.5)


   )
  (:ros-init ()
    (send-super :ros-init)

    )

  (:set-init-pose ()
    (setq *init-pose* (send self :update-robot-model))
    (ros::ros-info "Set init pose : ~A" *init-pose*)
    (send self :set-val 'init-pose *init-pose*)
    (dump-loadable-structure "init-pose.l" *init-pose*))
  (:load-init-pose ()
    (load "init-pose.l") ;; TODO from yaml
    (ros::ros-info "Loaded init pose is ~A" *init-pose*)
    (send self :set-val 'init-pose *init-pose*))
  (:init-pose ()
    (send self :load-init-pose)
    (send self :update-robot-model)
    (send robot-interface :angle-vector init-pose 3000)
    (send self :stop-grasp :wait t)
    (send robot-interface :wait-interpolation))


  (:angle-command-vector ()
       (send self :angle-vector (send robot :angle-vector)
             av-tm
             :min-time min-time :scale av-scale))


  (:start-grasp (&key (wait nil))
     (send robot-interface :start-grasp :wait wait))
  (:stop-grasp (&key (wait nil))
     (send robot-interface :stop-grasp :wait wait))

  ;; (:inverse-kinematics (target-coords &rest args)
  ;;     (send robot :arm :inverse-kinematics target-coords
  ;;            :rotation-axis t :move-target (send robot :arm :end-coords)
  ;;             :revert-if-fail nil :stop ik-stop-step
  ;;            :debug-view nil args))

  (:inverse-kinematics (target-coords)
      (send robot :arm :inverse-kinematics target-coords
             :rotation-axis t :move-target (send robot :arm :end-coords)
              :revert-if-fail nil :stop ik-stop-step))



  (:main-loop-step (&key (arm :arm) (imitation nil) (rot nil))
    (let ((ref-coords) (target-pos) (target-rpy) (robot-action) (robot-state))
      (send self :button-cb arm)
      ;; (send self :update-current-end-coords)
      (send *irtviewer* :draw-objects)
      (setf robot-state (send (send self :get-arm-val arm :current-end-coords) :copy-worldcoords)) ;; robot state

      ;; (print (list "robot-action-real" robot-action-real))


      (unless (send self :get-arm-val arm :enable)
        (send *ri* :stop-joint-velocity)
        )

      (if (send self :get-arm-val arm :enable)
        (progn
          (unless rot
            ;; (setf target-rpy (float-vector pi/2 pi/2 -pi/2)))
            (setf target-rpy (float-vector 0 0 0)))

          (if imitation
            (if rot
              (setf ref-coords (make-coords :pos (subseq robot-action-real 0 3) :rpy (subseq robot-action-real 3 6))) ;; rotational
              (setf ref-coords (make-coords :pos (subseq robot-action-real 0 3) :rpy target-rpy))) ;; fixed rpy

            (setf ref-coords (send (send self :get-target-coords arm :pos-scale 1.0) :copy-worldcoords)))

          (setf target-pos (send ref-coords :worldpos))
          (setf target-rpy (elt (rpy-angle (send ref-coords :worldrot)) 0))
          (unless rot
            ;; (setf target-rpy (float-vector pi/2 pi/2 -pi/2))
            (setf target-rpy (float-vector 0 0 0))
            )




          (setf target-coords (make-coords :pos target-pos :rpy target-rpy))


          (send self :inverse-kinematics target-coords)

          ;; (print target-coords)

          ;; (setq real-robot-av (send robot-interface :state :potentio-vector) )
          ;; (setq target-av (send robot :angle-vector))

          ;; (setq diff-av (v- target-av real-robot-av))
          ;; (print diff-av)

          (setf robot-action target-coords) ;; or joint angles

          (send self :robot-state-publisher robot-state rot) ;; state publisher
          (unless imitation
            (send self :robot-action-publisher robot-action rot))

          ;; (print (send robot-interface :gripper arm :position)) ;; gripper position
          ;; execute action


          (if imitation
              (setq gripper-action (if (> (elt robot-action-real 3) 0.5) 1 0))
              (setq gripper-action (send self :get-arm-val arm :grasp)))

          ;; TODO
          ;; (send self :angle-command-vector)
          ;; (send *ri* :angle-vector (send robot :angle-vector) av-tm :default-controller :min-time min-time :scale av-scale)
          ;; (send *ri* :angle-vector (send robot :angle-vector) av-tm )
          ;; (send *ri* :joint-velocity (scale 0.01 diff-av) )

          (setq end-coords-now (send (send robot :arm :end-coords) :worldpos))
          (setq target-coo (send target-coords :worldpos))

          (print end-coords-now)
          (print target-coo)

          (setq diff-coords (v- target-coo end-coords-now))

          ;; (send *ri* :joint-velocity (scale 0.01 diff-av) )
          (send *ri* :cartesian-velocity (scale 10.0 diff-coords) (float-vector 0 0 0))
          ;; (send *ri* :cartesian-velocity (float-vector 0.0 0.0 0.1) (float-vector 0 0 0))


            ;; (if (> gripper-action 0.5)
            ;;     ;; (send self :start-grasp :wait nil)
            ;;     ;; (send self :stop-grasp :wait nil))
            ;;     (send *ri* :start-grasp :wait nil)
            ;;     (send *ri* :stop-grasp :wait nil))

            ))))
  (:main-loop (&key (arm :rarm) (imitation nil) (rot nil))
    (send *ri* :angle-vector (send *kinova* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    ;; (send self :init-pose)

    (send self :update-current-end-coords)
    (if imitation
        (progn
          (send self :set-arm-val arm :enable t)
          (send self :init-pose)
          (setf robot-action-real (concatenate float-vector (send (send (send self :get-arm-val arm :current-end-coords) :copy-worldcoords) :worldpos ) (float-vector 0.0))) ;; 3 + 1
          (ros::subscribe "/eus_imitation/policy_action" eus_imitation::Float32MultiArrayStamped #'(lambda (msg) (setq robot-action-real (send msg :data))) 1)))

    ;; (send *ri* :speak "I'm ready to start." :wait nil :volume 0.6)
    (while (ros::ok)
      (ros::rate rate)
      (setq start-time (unix:gettimeofday))

      (send self :main-loop-step :arm arm :imitation imitation :rot rot)
      (ros::spin-once)
      (ros::sleep)

      (setq end-time (unix:gettimeofday))
      (setq wait-time (+ (* (- (elt end-time 0) (elt start-time 0)) 1000)
                          (/ (- (elt end-time 1) (elt start-time 1)) 1000.0)))
      (print wait-time)
      )))


  ;; ;; pick and place settings
  ;; ;; for single arm, fixed rotation
  (:update-robot-action () ;; when data collection
        (send self :set-val 'robot-action
              (concatenate float-vector
                  (send self :get-arm-val :rarm :target-end-pos)
                  ;; (float-vector (send self :get-arm-val :rarm :grasp))
                  (float-vector (send self :get-arm-val :rarm :target-gripper-pos))
                  ))) ;; (3 + 3 + 1) * 2 = 14

  (:update-robot-state ()
        (send self :set-val 'robot-state ;; you can define agent's state, proprioception
              (concatenate float-vector ;; concatenate all vectors
                  (send self :get-arm-val :rarm :current-end-pos)
                  (float-vector (send self :get-arm-val :rarm :gripper-pos))))) ;; (3 + 3 + 1) * 2 = 14

  (:parse-robot-action () ;; parse action from ros topic and execute it when imitation mode. define own action execution
      (let (action rarm-target-end-effector-coords rarm-target-end-effector-pos rarm-gripper-command)
        (setq action (send self :get-val 'robot-action))
        ;; parsing action
        (setq rarm-target-end-effector-pos (subseq action 0 3))
        (setq rarm-target-end-effector-rpy (float-vector pi/2 pi/2 -pi/2))
        (setq rarm-gripper-pos (elt action 3))
        (setq rarm-target-end-effector-coords (make-coords :pos rarm-target-end-effector-pos :rpy rarm-target-end-effector-rpy))
        ;; execute action
        (send self :inverse-kinematics :rarm rarm-target-end-effector-coords)
        (send self :angle-command-vector :rarm :torso nil)
        (send self :move-gripper :rarm rarm-gripper-pos :delta nil :wait nil)
        ;; (if (> rarm-gripper-command 0.01)
        ;;     (send self :start-grasp :rarm :wait nil)
        ;;     (send self :stop-grasp :rarm :wait nil))
        ))




  )


(provide :kinova-imitation-interface)
