(require :pr2-interface "package://pr2eus/pr2-interface.l") ;; pr2eus

(ros::load-ros-manifest "actionlib_msgs")
(ros::load-ros-manifest "pr2_teleop")
(ros::load-ros-manifest "moveit_msgs")
(ros::load-ros-manifest "sensor_msgs")
(ros::load-ros-manifest "std_msgs")
(ros::load-ros-manifest "std_srvs")

;; (ros::roseus-add-msgs "sensor_msgs")
;; (require :display-robot-state "package://jsk_interactive_marker/euslisp/display-robot-state.l")
;;
;;
;;

(setq init-lpos #f(800 188 1000))
(setq init-rpos #f(800 -188 1000))
(setq lpose (make-coords :pos init-lpos :rpy (float-vector 0 0 (deg2rad 90))))
(setq rpose (make-coords :pos init-rpos :rpy (float-vector 0 0 (deg2rad -90))))
(setq *test-target-coords* (make-coords :pos init-rpos :rpy (float-vector 0 0 (deg2rad -90))))

(defun joy-cb (joy-msg)
  ;; get cmd from joy
  (setq *linear-cmd* (subseq (send joy-msg :axes) 0 3))
  (setq *angular-cmd* (subseq (send joy-msg :axes) 3 6))
  (setq *button-cmd* (send joy-msg :buttons))
  )

;; (ros::subscribe "/spacenav/joy" sensor_msgs::Joy #'joy-cb) ;; 30 Hz

(defclass pr2-imitation-interface
  :super propertied-object
  :slots (robot
          robot-interface

          control-mode ;; RL, IL control or teleop for collecting demomstration
          robot-action ;; for RL or IL
          robot-action-type ;; EndEffectorPose (x,y,z,roll,pitch,yaw) or JointAngles (deg or rad) + Gripper (open or close)
          ;; robot-state-topic-name ;; for RL or IL state, can be list of multiple topics
          ;; ;; robot-obs-topic-name ;; for RL or IL observation, can be list of multiple topics

          reset-pose ;; initial pose for RL or IL
          loop-enable

          robot-action-topic-name
          pr2-imitation-topic-name

          pr2-imitation-reset-service-name
          pr2-imitation-reset-larm-service-name
          pr2-imitation-reset-rarm-service-name
          pr2-imitation-enable-service-name
          pr2-imitation-enable-larm-service-name
          pr2-imitation-enable-rarm-service-name
          pr2-imitation-disable-service-name
          pr2-imitation-disable-larm-service-name
          pr2-imitation-disable-rarm-service-name

          rate ;; control rate

          base-enable-p ;; TODO
          torse-enable-p ;; TODO
          larm-enable-p
          rarm-enable-p
          larm-grasping-p
          rarm-grasping-p
          larm-current-end-coords
          rarm-current-end-coords

          scale
          ik-stop-step
          av-tm
          av-scale
          min-time
          ;; arm-cb-solve-ik
          ))



(defmethod pr2-imitation-interface
  (:init  ()
    (send self :set-val 'robot *pr2*)
    (send self :set-val 'robot-interface *ri*)


    (send self :set-val 'rate 10)
    (send self :set-val 'av-tm 300)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'ik-stop-step 10)
    ;; for workspace
    ;; (send self :set-val 'pr2-imitation-reset-service-name "/robot_teleop/reset")
    ;; (send self :set-val 'pr2-imitation-reset-larm-service-name "/robot_teleop/larm/reset")
    ;; (send self :set-val 'pr2-imitation-reset-rarm-service-name "/robot_teleop/rarm/reset")
    ;; (send self :set-val 'pr2-imitation-enable-service-name "/robot_teleop/enable")
    ;; (send self :set-val 'pr2-imitation-enable-larm-service-name "/robot_teleop/larm/enable")
    ;; (send self :set-val 'pr2-imitation-enable-rarm-service-name "/robot_teleop/rarm/enable")
    ;; (send self :set-val 'pr2-imitation-disable-service-name "/robot_teleop/disable")
    ;; (send self :set-val 'pr2-imitation-disable-larm-service-name "/robot_teleop/larm/disable")
    ;; (send self :set-val 'pr2-imitation-disable-rarm-service-name "/robot_teleop/rarm/disable")
    ;; ;; (send self :set-val 'tfl (instance ros::transform-listener :init))

    self)
  (:ros-init ()
    t)




  ;; fundamental methods
;;   (:save-reset-pose ()
;;     (let* ((reset-pose (send robot-interface :state :potentio-vector)))
;;     (send robot :angle-vector reset-pose)
;;     (send self :set-val 'reset-pose reset-pose)
;;     (ros::ros-info "Reset pose saved as ~A" reset-pose)
;;     (setq *reset-pose* reset-pose) ;; TODO save yaml or something
;;     (dump-loadable-structure "reset-pose.l" *reset-pose*)))

;;   (:load-reset-pose ()
;;     (load "reset-pose.l") ;; TODO from yaml
;;     (ros::ros-info "Loaded reset pose is ~A" *reset-pose*)
;;     (send self :set-val 'reset-pose *reset-pose*))
)



(print "teleop loading complete")


;; (defun make-pr2-irtviewer (&key (no-window nil))
;;   (if (and no-window (string>= (car lisp-implementation-version) "9.28"))
;;     (make-irtviewer-no-window))
;;   (objects (list robot))
;;   (send *irtviewer* :change-background (float-vector 1 1 1))
;;   (send *irtviewer* :draw-floor 100)
;;   (send *irtviewer* :floor-color #f(0 0 0)))

;; (defun pr2-vive-init (&key (loop-enable t))
;;   (pr2-init)
;;   (setq *ti* (instance pr2-vive-interface :init))
;;   (make-pr2-irtviewer)
;;   (send *ti* :ros-init))

(defun main ()
  (pr2-init))

;; (defun init ()
;;   pr2-vive-init)
