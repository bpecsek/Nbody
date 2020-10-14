;;   The Computer Language Benchmarks Game
;;   http://benchmarksgame.alioth.debian.org/
;;;
;;; contributed by Patrick Frankenberger
;;; modified by Juho Snellman 2005-11-18
;;;   * About 40% speedup on SBCL, 90% speedup on CMUCL
;;;   * Represent a body as a DEFSTRUCT with (:TYPE VECTOR DOUBLE-FLOAT), a
;;;     not as a structure that contains vectors
;;;   * Inline APPLYFORCES
;;;   * Replace (/ DT DISTANCE DISTANCE DISTANCE) with
;;;     (/ DT (* DISTANCE DISTANCE DISTANCE)), as is done in the other
;;;     implementations of this test.
;;;   * Add a couple of declarations
;;;   * Heavily rewritten for style (represent system as a list instead of
;;;     an array to make the nested iterations over it less clumsy, use
;;;     INCF/DECF where appropriate, break very long lines, etc)
;;; modified by Marko Kocic 
;;;   * add optimization declarations
;;; modified by Bela Pecsek- 08/08/2020
;;;   * advance function written using AVX

(declaim (optimize (speed 3) (safety 0) (debug 0)))
(setf *block-compile-default* t)
(setf sb-ext:*inline-expansion-limit* 1000)
(sb-int:set-floating-point-modes :traps (list :divide-by-zero))
(declaim (sb-ext:muffle-conditions style-warning))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AVX inicialisation                                                      ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(load "avx-pack.lisp")

(in-package #:cl-user)

(define-constant +days-per-year+ 365.24d0)
(define-constant +solar-mass+ (* 4d0 pi pi))

(defstruct (body (:type (vector double-float))
                 (:conc-name nil)
                 (:constructor make-body (x y z vx vy vz mass)))
  x y z vx vy vz mass)

(deftype body () '(vector double-float 7))

(defparameter *jupiter*
  (make-body 4.84143144246472090d0
             -1.16032004402742839d0
             -1.03622044471123109d-1
             (* 1.66007664274403694d-3 +days-per-year+)
             (* 7.69901118419740425d-3 +days-per-year+)
             (* -6.90460016972063023d-5  +days-per-year+)
             (* 9.54791938424326609d-4 +solar-mass+)))

(defparameter *saturn*
  (make-body 8.34336671824457987d0
             4.12479856412430479d0
             -4.03523417114321381d-1
             (* -2.76742510726862411d-3 +days-per-year+)
             (* 4.99852801234917238d-3 +days-per-year+)
             (* 2.30417297573763929d-5 +days-per-year+)
             (* 2.85885980666130812d-4 +solar-mass+)))

(defparameter *uranus*
  (make-body 1.28943695621391310d1
             -1.51111514016986312d1
             -2.23307578892655734d-1
             (* 2.96460137564761618d-03 +days-per-year+)
             (* 2.37847173959480950d-03 +days-per-year+)
             (* -2.96589568540237556d-05 +days-per-year+)
             (* 4.36624404335156298d-05 +solar-mass+)))

(defparameter *neptune*
  (make-body 1.53796971148509165d+01
             -2.59193146099879641d+01
             1.79258772950371181d-01
             (* 2.68067772490389322d-03 +days-per-year+)
             (* 1.62824170038242295d-03 +days-per-year+)
             (* -9.51592254519715870d-05 +days-per-year+)
             (* 5.15138902046611451d-05 +solar-mass+)))

(defparameter *sun* (make-body 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 +solar-mass+))

(define-constant +bodies_count+ 5)
(define-constant +interactions_count+ (/ (* +bodies_count+ (1- +bodies_count+)) 2))
(define-constant +ric+ (+ +interactions_count+ (mod +interactions_count+ 4)))
(define-constant %1.0d4c (make-avx-double 1.0))
(define-constant +dt+  0.01d0)
(define-constant +recip_dt+ (/ 1.0d0 +dt+))

(deftype position_deltas () `(simple-array double-float (,+ric+)))
(deftype magnitudes () `(simple-array (double-float 0.0d0) (,+ric+)))

(defparameter *pdx* (make-array +ric+ :element-type 'double-float :initial-element 1.0d0))
(defparameter *pdy* (make-array +ric+ :element-type 'double-float :initial-element 1.0d0))
(defparameter *pdz* (make-array +ric+ :element-type 'double-float :initial-element 1.0d0))
(defparameter *mag* (make-array +ric+ :element-type '(double-float 0.0d0) :initial-element 0.0d0))
(declaim (type position_deltas *pdx* *pdy* *pdz*))
(declaim (type magnitudes *mag*))

(declaim (inline advance))
(defun advance (system times)
  (declare (list system)
	   (fixnum times))
  (let ((pdx *pdx*)
	(pdy *pdy*)
	(pdz *pdz*)
        (mag *mag*))
    (declare (position_deltas pdx pdy pdz)
             (magnitudes mag))
    (loop with i of-type fixnum = 0
	  for (a . rest) on system
	  do (dolist (b rest)
	       (declare (position_deltas pdx pdy pdz))
	       (setf (aref pdx i) (- (x a) (x b))
		     (aref pdy i) (- (y a) (y b))
		     (aref pdz i) (- (z a) (z b)))
	       (incf i)))
    (loop for i of-type fixnum below +ric+ by 4
	  do (let* ((%dx (d4ref pdx i))
		    (%dy (d4ref pdy i))
		    (%dz (d4ref pdz i))
		    (%posdelta_sq (d4+ (d4* %dx %dx)(d4* %dy %dy)(d4* %dz %dz)))
		    (%dist (d4sqrt %posdelta_sq))
		    (%mag (d4/ (d4* %dist %posdelta_sq))))
	       (setf (d4ref mag i) %mag))
	     (vzeroupper)) 
    (loop with i of-type fixnum = 0
	  for (a . rest) on system
	  do (dolist (b rest)
	       (declare (magnitudes mag)
			(position_deltas pdx pdy pdz))
	       (let* ((pdx-i (aref pdx i))
		      (pdy-i (aref pdy i))
		      (pdz-i (aref pdz i))
		      (mag-i (aref mag i))
		      (massmag-a (* (mass a) mag-i))  
		      (massmag-b (* (mass b) mag-i)))
		 (declare (double-float pdx-i pdy-i pdz-i)
			  ((double-float 0.0d0) massmag-a massmag-b))
		 (decf (vx a) (* pdx-i massmag-b)) 
	         (decf (vy a) (* pdy-i massmag-b))
	         (decf (vz a) (* pdz-i massmag-b))
	         (incf (vx b) (* pdx-i massmag-a))
	         (incf (vy b) (* pdy-i massmag-a)) 
	         (incf (vz b) (* pdz-i massmag-a)))	               
	       (incf i)))
    (dolist (a system)
      (declare (body a))
      (incf (x a) (vx a))
      (incf (y a) (vy a))
      (incf (z a) (vz a)))))

(defun energy (system)
  (declare (list system))
  (let ((e 0.0d0))
    (declare (double-float e))
    (loop for (a . rest) on system do
      (incf e (* 0.5d0 (mass a)
                 (+ (* (vx a) (vx a))
                    (* (vy a) (vy a))
                    (* (vz a) (vz a)))))
      (dolist (b rest)
        (let* ((dx (- (x a) (x b)))
               (dy (- (y a) (y b)))
               (dz (- (z a) (z b)))
               (dist (sqrt (+ (* dx dx) (* dy dy) (* dz dz)))))
	  (declare ((double-float 0.0d0) dist dx dy dz))
          (decf e (/ (* (mass a) (mass b)) dist)))))
    e))

(defun offset-momentum (system)
  (let ((px 0.0d0)
	(py 0.0d0)
	(pz 0.0d0)
        (sun (car system)))
    (declare (body sun)
	     ((double-float 0.0d0) px py pz))
    (dolist (p system)
      (incf px (* (vx p) (mass p)))
      (incf py (* (vy p) (mass p)))
      (incf pz (* (vz p) (mass p))))
    (setf (vx sun) (/ (- px) +solar-mass+)
          (vy sun) (/ (- py) +solar-mass+)
          (vz sun) (/ (- pz) +solar-mass+))
    nil))

(defun body-scale (system scale)
  (declare (list system)
	   ((double-float 0.0d0)  scale))
  (dolist (p system)
    (declare (body p))
    (setf (mass p) (* (mass p) (* scale scale))
	  (vx p) (* (vx p) scale)
	  (vy p) (* (vy p) scale)
	  (vz p) (* (vz p) scale))
    nil))

(defun nbody (n)
  (let ((system (list *sun* *jupiter* *saturn* *uranus* *neptune*)))
    (declare (list system)
	     (fixnum n))
    (offset-momentum system)
    (format t "~,9f~%" (energy system))
    (body-scale system +dt+)
    (loop repeat n do (advance system n))
    (body-scale system +recip_dt+)
    (format t "~,9f~%" (energy system))))

(defun main ()
 (let ((n (parse-integer (or (car (last sb-ext:*posix-argv*)) "50000000"))))
   (nbody n)))

