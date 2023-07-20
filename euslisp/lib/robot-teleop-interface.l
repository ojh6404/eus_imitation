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

(ros::roseus "pr2_test" :anonymous nil)

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

(ros::subscribe "/spacenav/joy" sensor_msgs::Joy #'joy-cb) ;; 30 Hz

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
  (:init  (&key (loop-enable t))
    (send self :set-val 'robot *pr2*)
    (send self :set-val 'robot-interface *ri*)


    (send self :set-val 'rate 10)
    (send self :set-val 'av-tm 300)
    (send self :set-val 'av-scale 5.0)
    (send self :set-val 'min-time 0.3)
    (send self :set-val 'ik-stop-step 10)
    ;; for workspace
    (send self :set-val 'pr2-imitation-reset-service-name "/robot_teleop/reset")
    (send self :set-val 'pr2-imitation-reset-larm-service-name "/robot_teleop/larm/reset")
    (send self :set-val 'pr2-imitation-reset-rarm-service-name "/robot_teleop/rarm/reset")
    (send self :set-val 'pr2-imitation-enable-service-name "/robot_teleop/enable")
    (send self :set-val 'pr2-imitation-enable-larm-service-name "/robot_teleop/larm/enable")
    (send self :set-val 'pr2-imitation-enable-rarm-service-name "/robot_teleop/rarm/enable")
    (send self :set-val 'pr2-imitation-disable-service-name "/robot_teleop/disable")
    (send self :set-val 'pr2-imitation-disable-larm-service-name "/robot_teleop/larm/disable")
    (send self :set-val 'pr2-imitation-disable-rarm-service-name "/robot_teleop/rarm/disable")
    ;; (send self :set-val 'tfl (instance ros::transform-listener :init))

    self)
  (:ros-init ()
    ;; (send self :reset :loop-enable loop-enable)
    ;; ;; (ros::subscribe larm-gripper-status-topic-name
    ;;                 actionlib_msgs::GoalStatusArray #'send self :gripper-status-cb :larm)
    ;; (ros::subscribe rarm-gripper-status-topic-name
    ;;                 actionlib_msgs::GoalStatusArray #'send self :gripper-status-cb :rarm)
    ;; (ros::advertise rarm-coords-visualize-topic-name geometry_msgs::PoseStamped 1)
    ;; (ros::advertise larm-grasp-visualize-topic-name std_msgs::Float32 1)
    ;; (ros::advertise rarm-grasp-visualize-topic-name std_msgs::Float32 1)
    ;; (ros::advertise robot-state-visualize-topic-name moveit_msgs::DisplayRobotState 1)
    ;; (ros::advertise pr2-imitation-status-topic-name pr2_teleop::TeleopStatusArray 1)
    ;; (ros::advertise-service pr2-imitation-reset-service-name std_srvs::Empty
    ;;                         #'send self :reset-service)
    ;; (ros::advertise-service pr2-imitation-reset-larm-service-name std_srvs::Empty
    ;;                         #'send self :reset-larm-service)
    ;; (ros::advertise-service pr2-imitation-reset-rarm-service-name std_srvs::Empty
    ;;                         #'send self :reset-rarm-service)
    ;; (ros::advertise-service pr2-imitation-enable-service-name std_srvs::Empty
    ;;                         #'send self :enable-service)
    ;; (ros::advertise-service pr2-imitation-enable-larm-service-name std_srvs::Empty
    ;;                         #'send self :enable-larm-service)
    ;; (ros::advertise-service pr2-imitation-enable-rarm-service-name std_srvs::Empty
    ;;                         #'send self :enable-rarm-service)
    ;; (ros::advertise-service pr2-imitation-disable-service-name std_srvs::Empty
    ;;                         #'send self :disable-service)
    ;; (ros::advertise-service pr2-imitation-disable-larm-service-name std_srvs::Empty
    ;;                         #'send self :disable-larm-service)
    ;; (ros::advertise-service pr2-imitation-disable-rarm-service-name std_srvs::Empty
    ;;                         #'send self :disable-rarm-service)
    t)




  ;; fundamental methods
  (:save-reset-pose ()
    (let* ((reset-pose (send robot-interface :state :potentio-vector)))
    (send robot :angle-vector reset-pose)
    (send self :set-val 'reset-pose reset-pose)
    (ros::ros-info "Reset pose saved as ~A" reset-pose)
    (setq *reset-pose* reset-pose) ;; TODO save yaml or something
    (dump-loadable-structure "reset-pose.l" *reset-pose*)))

  (:load-reset-pose ()
    (load "reset-pose.l") ;; TODO from yaml
    (ros::ros-info "Loaded reset pose is ~A" *reset-pose*)
    (send self :set-val 'reset-pose *reset-pose*))

  (:get-arm-controller (arm)
    (cond
      ((eq arm :larm) :larm-controller)
      ((eq arm :rarm) :rarm-controller)
      (t nil)))
  (:angle-vector (&rest args)
    (send* robot-interface :angle-vector args))
  (:angle-command-vector (arm &key (torso nil))
    (cond
      ((equal arm :arms)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :larm) 0
             :min-time min-time :scale av-scale)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :rarm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      ((equal arm :larm)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :larm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      ((equal arm :rarm)
       (send self :angle-vector (send robot :angle-vector)
             av-tm (send self :get-arm-controller :rarm) 0
             :min-time min-time :scale av-scale)
       (if torso
         (send self :angle-vector (send robot :angle-vector)
               torso-av-tm :torso-controller 0
               :min-time torso-min-time :scale torso-av-scale)))
      (t nil)))

  (:arm-ik (arm target-coords &rest args)
    (if (eq arm :arms)
      (send* robot arm :inverse-kinematics target-coords :rotation-axis (list t t)
             :move-target (list (send robot :larm :end-coords) (send robot :rarm :end-coords))
             :revert-if-fail nil :stop ik-stop-step :debug-view nil args)
      (send* robot arm :inverse-kinematics target-coords :rotation-axis t
             :move-target (send robot arm :end-coords)
             :revert-if-fail nil :stop ik-stop-step :debug-view nil args)))
  (:start-grasp (arm &key (wait nil))
     (send robot-interface :start-grasp arm :wait wait))
  (:stop-grasp (arm &key (wait nil))
     (send robot-interface :stop-grasp arm :wait wait))


  (:get-arm-val (arm name)
    (if (not (or (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :get-arm-val ~A ~A)" arm name))
        (return-from :get-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (find (read-from-string valname) (mapcar #'car (cdr (send self :slots))))
        (send self :get-val (intern valname)))
      ))
  (:set-arm-val (arm name val)
    (if (not (or (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :set-arm-val: ~A ~A ~A)" arm name val))
        (return-from :set-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (send self :set-val (intern valname) val)
      ))
  (:update-current-end-coords ()
     (let ((av (send robot :angle-vector)))
       (send robot :angle-vector
             (send robot-interface :state :potentio-vector))
       (send self :set-arm-val :larm :current-end-coords
             (send robot :larm :end-coords :copy-worldcoords))
       (send self :set-arm-val :rarm :current-end-coords
             (send robot :rarm :end-coords :copy-worldcoords))
       (send robot :angle-vector av)))

  (:control-loop-step (&key (head t) (torso t) (safe-arm t) (mirror nil)
                         (enable-button :menu) (gripper-button :trigger) (visualize t))
    (let (visualize-arm-coords)
      (send self :enable-cb :mirror mirror :button enable-button)
      (setq visualize-arm-coords
            (send self :motion-cb :head head :torso torso :safe-arm safe-arm :mirror mirror
                  :gripper-button gripper-button))
      (send self :update-current-end-coords)))
  (:control-loop (&rest args)
    (ros::rate rate)
    (send self :update-current-end-coords)
    (send *ri* :speak "I'm ready to start control loop." :wait nil :volume 0.6)
    (while (ros::ok)
      (ros::spin-once)
      (send robot-interface :cancel-angle-vector
                :controller-type (send self :get-arm-controller :rarm))
      ;; (setq *test-target-coords* (make-coords :pos (v+ (scale 50 *linear-cmd*) (send current-arm-coords :pos)) :rpy (float-vector 0 0 (deg2rad -90))))
      ;; (if loop-enable (send* self :control-loop-step args))
      (setq *robot-action* (make-coords :pos (v+ (scale 70 *linear-cmd*) (send (send self :get-arm-val :rarm :current-end-coords) :pos)) :rpy (float-vector 0 0 (deg2rad -90))))
      (send self :arm-ik :rarm *robot-action*)

      ;; (send *ri* :angle-vector (send *pr2* :angle-vector) :rarm-controller 0 :min-time 0.5 :scale 1.0)
      ;; (send *ri* :angle-vector (send *pr2* :angle-vector) :fast nil 0 :min-time 1.0 :scale 1)
      ;; (send *ri* :angle-vector (send *pr2* :angle-vector) 50 :rarm-controller)
      ;; (send self :angle-command-vector :rarm :av-tm 300 :av-scale 5.0 :min-time 0.3)
      ;; (send self :angle-command-vector :rarm)
      (setq av-tm 80)
      (setq min-time 0.3)
      (setq av-scale 5.0)
      (send self :angle-vector (send *pr2* :angle-vector)
             av-tm (send self :get-arm-controller :rarm) 0
             :min-time min-time :scale av-scale)
      (send self :update-current-end-coords)
      (send *irtviewer* :draw-objects)
      (ros::sleep)))
  )

"Send joint angle to robot, this method returns immediately, so use :wait-interpolation to block until the motion stops.
- av : joint angle vector [deg]
- tm : (time to goal in [msec])
   if designated tm is faster than fastest speed, use fastest speed
   if not specified, it will use 1/scale of the fastest speed .
   if :fast is specified use fastest speed calculated from max speed
- ctype : controller method name
- start-time : time to start moving
- scale : if tm is not specified, it will use 1/scale of the fastest speed
- min-time : minimum time for time to goal
- end-coords-interpolation : set t if you want to move robot in cartesian space interpolation
- end-coords-interpolation-steps : number of divisions when interpolating end-coords
- minjerk-interpolation : set t if you want to move robot with minjerk interpolation
"


(print "teleop loading complete")


(defun make-pr2-irtviewer (&key (no-window nil))
  ;; (if (and no-window (string>= (car lisp-implementation-version) "9.28"))
  ;;   (make-irtviewer-no-window))
  (objects (list *pr2* rpose))
  ;; (send *irtviewer* :change-background (float-vector 1 1 1))
  ;; (send *irtviewer* :draw-floor 100)
  ;; (send *irtviewer* :floor-color #f(0 0 0)))
  )
(defun pr2-vive-init (&key (loop-enable t))
  (pr2-init)
  (setq *ti* (instance pr2-imitation-interface :init))
  (make-pr2-irtviewer)
  (send *ti* :ros-init))


(defun main ()
  (pr2-vive-init))
