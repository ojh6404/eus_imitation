#!/usr/bin/env roseus

(defclass interface-base
  :super propertied-object
  :slots ())

(defmethod interface-base
  (:get-arm-val (arm name)
    (if (not (or (eq arm :arms) (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :get-arm-val ~A ~A)" arm name))
        (return-from :get-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (eq arm :arms)
        (progn
          (if (and (eq (send self :get-arm-val :larm name) nil)  (eq (send self :get-arm-val :rarm name) nil))
            (return-from :get-arm-val nil)
            (progn
              (return-from :get-arm-val (list (send self :get-arm-val :larm name)
                                              (send self :get-arm-val :rarm name)))))))
      (if (find (read-from-string valname) (mapcar #'car (cdr (send self :slots))))
        (send self :get-val (intern valname)))))
  (:set-arm-val (arm name val)
    (if (not (or (eq arm :arms) (eq arm :larm) (eq arm :rarm)))
      (progn
        (ros::ros-error (format nil "Invalid args: (send self :set-arm-val: ~A ~A ~A)" arm name val))
        (return-from :set-arm-val nil)))
    (let ((valname (format nil "~A-~A" (string-upcase arm) (string-upcase name))))
      (if (eq arm :arms)
        (progn
          (if (listp val)
            (progn
              (send self :set-arm-val :larm name (first val))
              (send self :set-arm-val :rarm name (second val)))
            (progn
              (send self :set-arm-val :larm name val)
              (send self :set-arm-val :rarm name val))))
        (send self :set-val (intern valname) val)))))

(provide :interface-base)
