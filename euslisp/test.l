#!/usr/bin/env roseus

(require :pr2-interface "package://pr2eus/pr2-interface.l")
(ros::roseus "test_node")
(ros::roseus-add-msgs "jsk_recognition_msgs")

(setq *bbox-msg*
      (one-shot-subscribe "/tracking_ros/segmentor/output/boxes"
			  jsk_recognition_msgs::BoundingBoxArray
			  :timeout (* 10 1000)))


(let*
  ((boxes (send *bbox-msg* :boxes))
   (box (if boxes (car boxes))))
  (if box
    (progn
      (setq *target-coords* (ros::tf-pose->coords (send box :pose)))
      ;; (send *target-coords* :rotate (/ -pi 2) :y)
      ;; (send *target-coords* :rotate pi :x)
      (setq *target-cube* (make-cube (* 1000 (send box :dimensions :z))
                                     (* 1000 (send box :dimensions :y))
                                     (* 1000 (send box :dimensions :x))))
      (send *target-cube* :newcoords (send *target-coords* :copy-worldcoords))
      )
    )
  )

(pr2-init)
(print *target-coords*)
(objects (list *pr2* *target-coords* *target-cube*))
