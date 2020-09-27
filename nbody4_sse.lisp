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
;;; modified by Bela Pecsek
;;;   * advance function written with SSE support based on GCC#8
;;;   * SSE operators re-written based on code of Numericals package of 
;;;     Shubhamkar Ayare with compiler-macros
;;;   * code modification to use scaling as in GCC#2 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; SSE inicialisation                                                        ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(declaim (optimize speed (safety 0) (space 0) (debug 0)))
(in-package :sb-vm)
(eval-when (:compile-toplevel :load-toplevel :execute)
   (defmacro define-double-binary-vop-operation (name sse-operation)
     `(eval-when (:compile-toplevel :load-toplevel :execute)
        (defknown (,name) ((simd-pack double-float) (simd-pack double-float))
         (simd-pack double-float)
         (movable flushable always-translatable)
         :overwrite-fndb-silently t)
        (define-vop (,name)
          (:translate ,name)
          (:policy :fast-safe)
          (:args (a :scs (double-sse-reg))
                 (b :scs (double-sse-reg) :to :save))
          (:arg-types simd-pack-double simd-pack-double)
          (:results (dest :scs (double-sse-reg)))
          (:result-types simd-pack-double)
          (:generator 1 (inst ,sse-operation dest a b)))))
   (declaim (inline %d2+ %d2* %d2/))
   (define-double-binary-vop-operation %d2+ vaddpd)
   (define-double-binary-vop-operation %d2* vmulpd)
   (define-double-binary-vop-operation %d2/ vdivpd)

  (defmacro define-double-unary-vop-operation (name sse-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defknown (,name) ((simd-pack double-float)) (simd-pack double-float)
           (movable flushable always-translatable)
         :overwrite-fndb-silently t)
       (define-vop (,name)
         (:translate ,name)
         (:policy :fast-safe)
         (:args (a :scs (double-sse-reg)))
         (:arg-types simd-pack-double)
         (:results (dest :scs (double-sse-reg)))
         (:result-types simd-pack-double)
         (:generator 1 (inst ,sse-operation dest a)))))
  (declaim (inline %d2sqrt))
  (define-double-unary-vop-operation %d2sqrt vsqrtpd)

  (declaim (inline %d2-ref %d2-set))
  (defknown %d2-ref ((simple-array double-float (*))
                     (integer 0 #.most-positive-fixnum))
      (simd-pack double-float)
      (movable foldable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d2-ref)
    (:translate %d2-ref)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg)))
    (:arg-types simple-array-double-float
                tagged-num)
    (:results (dest :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:policy :fast-safe)
    (:generator 4 (inst movapd dest (float-ref-ea v i 0 8 :scale 4))))

  (defknown %d2-set ((simple-array double-float (*))
                     (integer 0 #.most-positive-fixnum)
                     (simd-pack double-float)) (simd-pack double-float)
      (always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d2-set)
    (:translate %d2-set)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg))
           (x :scs (double-sse-reg) :to :save))
    (:arg-types simple-array-double-float
                tagged-num
                simd-pack-double)
    (:policy :fast-safe)
    (:generator 4 (inst movapd (float-ref-ea v i 0 8 :scale 4) x))))

(in-package :cl-user)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (declaim (inline %make-sse-double))
  (defun %make-sse-double (a b)
    (declare (optimize speed (safety 0) (debug 0))
	     (type double-float a b))
    (truly-the (simd-pack double-float) (sb-vm::%make-simd-pack-double a b)))

  (defmacro define-constant (name value &optional doc)
    `(defconstant ,name (if (boundp ',name) (symbol-value ',name) ,value)
       ,@(when doc (list doc))))
  (define-constant %0.0d2c (%make-sse-double 0.0d0 0.0d0))
  (define-constant %1.0d2c (%make-sse-double 1.0d0 1.0d0))

  (declaim (inline d2+))
  (defun d2+ (&rest args)
    (declare (optimize speed (safety 0) (debug 0) (space 0)))
    (cond ((null args) %0.0d2c)
          ((null (cdr args)) (car args))
          ((null (cddr args)) (truly-the (simd-pack double-float)
                               (sb-vm::%d2+ (car args) (cadr args))))
          (t (truly-the (simd-pack double-float)
              (sb-vm::%d2+ (car args) (apply #'d2+ (cdr args)))))))
  (define-compiler-macro d2+ (&whole whole &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
     (let ((args (case (car whole)
                  (apply (nconc (butlast (cdr args))
                          (car (last (cdr args)))))
                  (funcall args)
                  (t args))))
          (cond ((null args) %0.0d2c)
           ((null (cdr args)) (car args))
           ((null (cddr args)) `(sb-vm::%d2+ ,(car args) ,(cadr args)))
           (t `(sb-vm::%d2+ ,(car args)
                ,(funcall (compiler-macro-function 'd2+)
                          `(funcall #'d2+ ,@(cdr args)) env))))) whole))
  (declaim (inline d2*))
  (defun d2* (&rest args)
    (declare (optimize speed (safety 0) (debug 0) (space 0)))
    (cond ((null args) %1.0d2c)
          ((null (cdr args)) (car args))
          ((null (cddr args)) (truly-the (simd-pack double-float)
                               (sb-vm::%d2* (car args) (cadr args))))
          (t (truly-the (simd-pack double-float)
              (sb-vm::%d2* (car args) (apply 'd2* (cdr args)))))))
  (define-compiler-macro d2* (&whole whole &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
     (let ((args (case (car whole)
                       (apply (nconc (butlast (cdr args))
                               (car (last (cdr args)))))
                       (funcall args)
                       (t args))))
          (cond ((null args) %1.0d2c)
           ((null (cdr args)) (car args))
           ((null (cddr args)) `(sb-vm::%d2* ,(car args) ,(cadr args)))
           (t `(sb-vm::%d2* ,(car args)
                ,(funcall (compiler-macro-function 'd2*)
                          `(funcall #'d2* ,@(cdr args)) env))))) whole))
  (declaim (inline d2/))
  (defun d2/ (arg &rest args)
    (declare (optimize speed (safety 0) (debug 0) (space 0)))
    (cond ((null args) (sb-vm::%d2/ %1.0d2c arg))
          ((null (cdr args)) (sb-vm::%d2/ arg (car args)))
          (t (truly-the (simd-pack double-float)
              (apply #'d2/ (sb-vm::%d2/ arg (car args))
                     (cdr args))))))
  (define-compiler-macro d2/ (&whole whole arg &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
     (cond ((null args) `(sb-vm::%d2/ ,%1.0d2c ,arg))
           ((null (cdr args)) `(sb-vm::%d2/ ,arg ,(car args)))
           (t (funcall (compiler-macro-function 'd2/)
               `(funcall #'d2/ (sb-vm::%d2/ ,arg ,(car args))
                              ,@(cdr args)) env))) whole))

  (defmacro define-double-unary-operation (name sse-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (declaim (inline ,name))
       (defun ,name (x)
        (declare (optimize speed (safety 0) (debug 0) (space 0))
           (type (simd-pack double-float) x))
        (truly-the (simd-pack double-float) (,sse-operation x)))))
  (define-double-unary-operation d2sqrt sb-vm::%d2sqrt)

  (declaim (inline d2-ref))
  (defun d2-ref (v i)
    (declare (optimize speed (safety 0) (debug 0) (space 0))
       (type (simple-array double-float (*)) v)
       (type (integer 0 #.most-positive-fixnum) i))
    (truly-the (simd-pack double-float) (sb-vm::%d2-ref v i)))
  
  (declaim (inline (setf d2-ref)))
  (defun (setf d2-ref) (new-value v i)
    (declare (optimize speed (safety 0) (debug 0) (space 0))
       (type (simple-array double-float (*)) v)
       (type (integer 0 #.most-positive-fixnum) i)
       (type (simd-pack double-float) new-value))
    (truly-the (simd-pack double-float) (sb-vm::%d2-set v i new-value))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(in-package :cl-user)

(defconstant +DAYS-PER-YEAR+ 365.24d0)
(defconstant +SOLAR-MASS+ (* 4d0 pi pi))

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

(defparameter *sun* (make-body 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
                     +solar-mass+))
(defparameter *system* (list *sun* *jupiter* *saturn* *uranus* *neptune*))

;; Declarations and iteration of 'advance' function is based on GCC#8 and GCC#2
;; Figure out how many total different interactions there are between each
;; body and every other body. Some of the calculations for these
;; interactions will be calculated two at a time by using x86 SSE
;; instructions and because of that it will also be useful to have a
;; ROUNDED_INTERACTIONS_COUNT that is equal to the next highest even number
;; which is equal to or greater than INTERACTIONS_COUNT.
(define-constant +BODIES_COUNT+ (length *system*))
(define-constant +INTERACTIONS_COUNT+ (/ (* +BODIES_COUNT+
                                          (1- +BODIES_COUNT+)) 2))
(define-constant +ROUNDED_INTERACTIONS_COUNT+ (+ +INTERACTIONS_COUNT+
                                               (mod +INTERACTIONS_COUNT+ 2)))
;; Based on GCC #2
(define-constant +DT+ 0.01d0)
(define-constant +RECIP_DT+ (/ +dt+))
;; Type declarations
(deftype position_deltas () `(simple-array double-float
                              (,+ROUNDED_INTERACTIONS_COUNT+)))
(deftype magnitudes () `(simple-array double-float
                         (,+ROUNDED_INTERACTIONS_COUNT+)))

;; Based on GCC #8 and GCC #2
;; Advances with timestem dt = 1.0d0
;; Advance all the bodies in the system by one timestep. Calculate the
;; interactions between all the bodies, update each body's velocity based on
;; those interactions, and update each body's position by the distance it
;; travels in a timestep of 1.0d0 at it's updated velocity.
;; 3 simple arrays are used to stope position delta components for speed
;; Force magnitudes are stores in array magnitude
(defun advance (system n)
  (declare (type list system)
	   (type fixnum n))
  (let ((position_Deltas-x (make-array +ROUNDED_INTERACTIONS_COUNT+
                            :element-type 'double-float :initial-element 1.0d0))
        (position_Deltas-y (make-array +ROUNDED_INTERACTIONS_COUNT+
                            :element-type 'double-float :initial-element 1.0d0))
        (position_Deltas-z (make-array +ROUNDED_INTERACTIONS_COUNT+
                            :element-type 'double-float :initial-element 1.0d0))
        (magnitudes (make-array +ROUNDED_INTERACTIONS_COUNT+
                     :element-type 'double-float :initial-element 1.0d0)))
    (flet ((position-deltas ()
       ;; Calculate the position_Deltas between the bodies for each
       ;; interaction.
            (loop with i of-type fixnum = 0 for (a . rest) on system 
             do (dolist (b rest)
		  ;; (declare (type position_deltas pdx pdy pdz))
                 (setf (aref position_Deltas-x i) (- (x a) (x b))
                       (aref position_Deltas-y i) (- (y a) (y b))
                       (aref position_Deltas-z i) (- (z a) (z b)))
                 (incf i))))
           (force-magnitudes ()
	     ;; Calculate the magnitudes of force between the bodies for each
	     ;; interaction. This loop processes two interactions at a time
	     ;; which is why ROUNDED_INTERACTIONS_COUNT/2 iterations are done.
             (loop for i of-type fixnum below +ROUNDED_INTERACTIONS_COUNT+ by 2
              do (let* ((pdx (d2-ref position_Deltas-x i))
                        (pdy (d2-ref position_Deltas-y i))
                        (pdz (d2-ref position_Deltas-z i))
                        (distance_Squared (d2+ (d2* pdx pdx) (d2* pdy pdy)
                                           (d2* pdz pdz)))
                        (magnitude (d2/ (d2* (d2sqrt distance_Squared)
                                         distance_Squared))))
                  (setf (d2-ref magnitudes i) magnitude))))
           (velocities ()
	     ;; Use the calculated magnitudes of force to update the velocities
	     ;; for all of the bodies.
             (loop with i of-type fixnum = 0 for (a . rest) on system
              do (dolist (b rest)
                  (let* ((pdx-i (aref position_Deltas-x i))
                         (pdy-i (aref position_Deltas-y i))
                         (pdz-i (aref position_Deltas-z i))
                         (mag-i (aref magnitudes i))
			 ;; Precompute products of mass and magnitude
			 ;; to be reused a couple of times
                         (mass*mag-a (* (mass a) mag-i))  
                         (mass*mag-b (* (mass b) mag-i)))
                    (decf (vx a) (* pdx-i mass*mag-b)) 
                    (decf (vy a) (* pdy-i mass*mag-b))
                    (decf (vz a) (* pdz-i mass*mag-b))
                    (incf (vx b) (* pdx-i mass*mag-a))
                    (incf (vy b) (* pdy-i mass*mag-a)) 
                    (incf (vz b) (* pdz-i mass*mag-a)))                 
                  (incf i))))
           (positions ()
	     ;; Use the updated velocities to update the positions for all
	     ;; bodies.
             (dolist (a system)
               (incf (x a) (vx a))
               (incf (y a) (vy a))
               (incf (z a) (vz a)))))
      (declare (inline position-deltas force-magnitudes velocities positions))
      (loop repeat n do (progn (position-deltas)
			       (force-magnitudes)
                               (velocities)
			       (positions))))))

;; Output the total energy of the system.
(defun output_Energy (system)
    (let ((e 0.0d0))
      (declare (type double-float e))
      (loop for (a . rest) on system do
	;; Add the kinetic energy for each body.
        (incf e (* 0.5d0 (mass a)
                   (+ (* (vx a) (vx a))
                      (* (vy a) (vy a))
                      (* (vz a) (vz a)))))
        (dolist (b rest)
	  ;; Add the potential energy between this body and every other bodies
          (let* ((dx (- (x a) (x b)))
                 (dy (- (y a) (y b)))
                 (dz (- (z a) (z b)))
                 (dist (sqrt (+ (* dx dx) (* dy dy) (* dz dz)))))
            (decf e (/ (the double-float (* (mass a) (mass b))) dist)))))
      (format t "~,9f~%" e))) ;; Output the total energy of the system.

;; Calculate the momentum of each body and conserve momentum of the system by
;; adding to the Sun's velocity the appropriate opposite velocity needed in
;; order to offset that body's momentum.
(defun offset-momentum (system)
  (let ((px 0.0d0)
        (py 0.0d0)
        (pz 0.0d0)
        (sun (car system)))
    (dolist (p system)
      (incf px (* (vx p) (mass p)))
      (incf py (* (vy p) (mass p)))
      (incf pz (* (vz p) (mass p))))
    (setf (vx sun) (/ (- px) +solar-mass+)
          (vy sun) (/ (- py) +solar-mass+)
          (vz sun) (/ (- pz) +solar-mass+))
    nil))

;; Rescale certain properties of bodies. That allows doing
;; consequential advances as if dt were equal to 1.0d0.
;; When all advances done, rescale bodies back to obtain correct energy.
(defun scale_bodies (system scale)
  (declare (type list system)
     (type double-float scale))
  (dolist (p system)
    (declare (type body p))
    (setf (mass p) (* (mass p) (* scale scale))
     (vx p) (* (vx p) scale)
     (vy p) (* (vy p) scale)
     (vz p) (* (vz p) scale))
    nil))

(defun nbody (n)
  (let ((system *system*))
    (offset-momentum system)         ;; Output initial energy of the system
    (output_Energy system)           ;; Scale bodies to use time step of 1.0d0
    (scale_bodies system +DT+)       ;; Advance system with time step of 1.0d0
    (advance system n)               ;; Advance system n times
    (scale_bodies system +RECIP_DT+) ;; Rescale bodies
    (output_Energy system)))         ;; Output final energy of the system

(defun main (&optional n-supplied)
 (let* ((args sb-ext:*posix-argv*)
        (n (or n-supplied (parse-integer (or (second args) "50000000")))))
   (nbody n)))

