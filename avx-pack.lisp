;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Setting up AVX2 functionality                                             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(declaim (optimize (speed 3) (safety 0) (space 0) (debug 0)))
(setf sb-ext:*efficiency-note-cost-threshold* 1)
(setf sb-ext:*efficiency-note-limit* 8)
(setf *block-compile-default* t)
(setf sb-ext:*inline-expansion-limit* 0)
(sb-int:set-floating-point-modes :traps (list :divide-by-zero))
(declaim (sb-ext:muffle-conditions style-warning))

(in-package #:sb-vm)

(deftype %d4  () '(simd-pack-256 double-float))
(deftype %d4+ () '(simd-pack-256 (double-float 0.0d0)))
(deftype %s8  () '(simd-pack-256 single-float))
(deftype %s8+ () '(simd-pack-256 (single-float 0.0)))
(deftype %s4  () '(simd-pack single-float))
(deftype %s4+ () '(simd-pack (single-float 0.0)))
(deftype %d2  () '(simd-pack double-float))
(deftype %d2+ () '(simd-pack (double-float 0.0d0)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defmacro define-double-binary-vop-operation (name avx2-operation cost)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defknown (,name) (%d4 %d4) %d4
           (movable flushable always-translatable)
         :overwrite-fndb-silently t)
       (define-vop (,name)
         (:translate ,name)
         (:policy :fast-safe)
         (:args (a :scs (double-avx2-reg))
                (b :scs (double-avx2-reg)))
         (:arg-types simd-pack-256-double simd-pack-256-double)
         (:results (result :scs (double-avx2-reg)))
         (:result-types simd-pack-256-double)
         (:generator ,cost ;; what should be the cost?
                     (inst ,avx2-operation result a b)))))

    (defmacro define-single-binary-vop-operation (name avx2-operation cost)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defknown (,name) (%s8 %s8) %s8
           (movable flushable always-translatable)
         :overwrite-fndb-silently t)
       (define-vop (,name)
         (:translate ,name)
         (:policy :fast-safe)
         (:args (a :scs (single-avx2-reg))
                (b :scs (single-avx2-reg)))
         (:arg-types simd-pack-256-single simd-pack-256-single)
         (:results (result :scs (single-avx2-reg)))
         (:result-types simd-pack-256-single)
         (:generator ,cost ;; what should be the cost?
                     (inst ,avx2-operation result a b)))))

  (defmacro define-double-unary-vop-operation (name avx2-operation cost)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (defknown (,name) (%d4) %d4
                 (movable flushable always-translatable)
                 :overwrite-fndb-silently t)
       (define-vop (,name)
           (:translate ,name)
         (:policy :fast-safe)
         (:args (a :scs (double-avx2-reg)))
         (:arg-types simd-pack-256-double)
         (:results (result :scs (double-avx2-reg)))
         (:result-types simd-pack-256-double)
         (:generator ,cost ;; TODO: what should be the cost?
                     (inst ,avx2-operation result a)))))

    (defmacro define-single-unary-vop-operation (name avx2-operation cost)
      `(eval-when (:compile-toplevel :load-toplevel :execute)
         (defknown (,name) (%s8) %s8
             (movable flushable always-translatable)
           :overwrite-fndb-silently t)
         (define-vop (,name)
           (:translate ,name)
           (:policy :fast-safe)
           (:args (a :scs (single-avx2-reg)))
           (:arg-types simd-pack-256-single)
           (:results (result :scs (single-avx2-reg)))
           (:result-types simd-pack-256-single)
           (:generator ,cost ;; TODO: what should be the cost?
                       (inst ,avx2-operation result a)))))

    (defmacro define-d2s-conv-unary-vop-operation (name avx2-operation cost)
      `(eval-when (:compile-toplevel :load-toplevel :execute)
         (defknown (,name) (%d4) %s8
             (movable flushable always-translatable)
           :overwrite-fndb-silently t)
         (define-vop (,name)
           (:translate ,name)
           (:policy :fast-safe)
           (:args (a :scs (double-avx2-reg)))
           (:arg-types simd-pack-256-double)
           (:results (result :scs (single-avx2-reg)))
           (:result-types simd-pack-256-single)
           (:generator ,cost ;; TODO: what should be the cost?
                       (inst ,avx2-operation result a)))))

    (defmacro define-s2d-conv-unary-vop-operation (name avx2-operation cost)
      `(eval-when (:compile-toplevel :load-toplevel :execute)
         (defknown (,name) (%s8) %d4
             (movable flushable always-translatable)
           :overwrite-fndb-silently t)
         (define-vop (,name)
           (:translate ,name)
           (:policy :fast-safe)
           (:args (a :scs (single-avx2-reg)))
           (:arg-types simd-pack-256-single)
           (:results (result :scs (double-avx2-reg)))
           (:result-types simd-pack-256-double)
           (:generator ,cost ;; TODO: what should be the cost?
                       (inst ,avx2-operation result a)))))
  
  (define-double-binary-vop-operation %d4+ vaddpd 1)
  (define-double-binary-vop-operation %d4- vsubpd 1)
  (define-double-binary-vop-operation %d4* vmulpd 1)
  (define-double-binary-vop-operation %d4/ vdivpd 1)
  (define-single-binary-vop-operation %s8+ vaddps 1)
  (define-single-binary-vop-operation %s8- vsubps 1)
  (define-single-binary-vop-operation %s8* vmulps 1)
  (define-single-binary-vop-operation %s8/ vdivps 1)
  (define-double-unary-vop-operation %d4sqrt vsqrtpd 2)
  (define-single-unary-vop-operation %s4rcpps vrcpps 2)
  (define-single-unary-vop-operation %s4rsqrt vrsqrtps 2)
  (define-d2s-conv-unary-vop-operation %d4pd2ps vcvtpd2ps 1)
  (define-s2d-conv-unary-vop-operation %s4ps2pd vcvtps2pd 1)
  
  (defknown %d4ref ((simple-array double-float (*))
                    (integer 0 #.most-positive-fixnum)) %d4
      (movable foldable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4ref)
    (:translate %d4ref)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg)))
    (:arg-types simple-array-double-float
                tagged-num)
    (:results (result :scs (double-avx2-reg)))
    (:result-types simd-pack-256-double)
    (:policy :fast-safe)
    (:generator 1 (inst vmovupd result (float-ref-ea v i 0 0
			:scale (ash 8 (- n-fixnum-tag-bits))))))

   (defknown %s8ref ((simple-array single-float (*))
                    (integer 0 #.most-positive-fixnum)) %s8
      (movable foldable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%s8ref)
    (:translate %s8ref)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg)))
    (:arg-types simple-array-single-float
                tagged-num)
    (:results (result :scs (single-avx2-reg)))
    (:result-types simd-pack-256-single)
    (:policy :fast-safe)
    (:generator 1 (inst vmovupd result (float-ref-ea v i 0 0
			:scale (ash 8 (- n-fixnum-tag-bits))))))

  (defknown %d4set ((simple-array double-float (*))
                    (integer 0 #.most-positive-fixnum)
                   %d4) %d4
      (always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4set)
    (:translate %d4set)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg))
           (x :scs (double-avx2-reg) :target result))
    (:arg-types simple-array-double-float
                tagged-num
                simd-pack-256-double)
    (:results (result :scs (double-avx2-reg) :from (:argument 2)))
    (:result-types simd-pack-256-double)
    (:policy :fast-safe)
    (:generator 1 (inst vmovups (float-ref-ea v i 0 0
		        :scale (ash 8 (- n-fixnum-tag-bits))) x)))

  (defknown %s8set ((simple-array single-float (*))
                    (integer 0 #.most-positive-fixnum)
                   %s8) %s8
      (always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%s8set)
    (:translate %s8set)
    (:args (v :scs (descriptor-reg))
           (i :scs (any-reg))
           (x :scs (single-avx2-reg) :target result))
    (:arg-types simple-array-single-float
                tagged-num
                simd-pack-256-single)
    (:results (result :scs (single-avx2-reg) :from (:argument 2)))
    (:result-types simd-pack-256-single)
    (:policy :fast-safe)
    (:generator 1 (inst vmovups (float-ref-ea v i 0 0
 		        :scale (ash 8 (- n-fixnum-tag-bits))) x)))
  
  (defknown (%d4rec) (%d4 %d4+) %d4
      (movable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4rec)
    (:translate %d4rec)
    (:policy :fast-safe)
    (:args (x :scs (double-avx2-reg))
	   (two :scs (double-avx2-reg)))    
    (:arg-types simd-pack-256-double simd-pack-256-double)
    (:temporary (:sc double-avx2-reg) %ymm0)
    (:temporary (:sc double-avx2-reg) %ymm1)
    (:temporary (:sc double-avx2-reg) %ymm2)
    (:results (result :scs (double-avx2-reg)))
    (:result-types simd-pack-256-double)
    (:generator 8 ;; what should be the cost?
                (inst vmovapd %ymm0 x)
		(inst vcvtpd2ps %ymm0 %ymm0)
		(inst vrcpps %ymm0 %ymm0)
		(inst vcvtps2pd %ymm0 %ymm0)
		(inst vmulpd %ymm2 two %ymm0)
		(inst vmulpd %ymm0 %ymm0 %ymm0)
		(inst vmulpd %ymm1 x %ymm0)
		(inst vsubpd %ymm0 %ymm2 %ymm1)
		(inst vmulpd %ymm2 two %ymm0)
		(inst vmulpd %ymm0 %ymm0 %ymm0)
		(inst vmulpd %ymm1 x %ymm0)
		(inst vsubpd result %ymm2 %ymm1)))

  (defknown (%d4rsqrt) (%d4 %d4+ %d4+) %d4
      (movable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4rsqrt)
    (:translate %d4rsqrt)
    (:policy :fast-safe)
    (:args (x :scs (double-avx2-reg))
	   (half :scs (double-avx2-reg))
	   (threehalfs :scs (double-avx2-reg)))    
    (:arg-types simd-pack-256-double simd-pack-256-double simd-pack-256-double)
    (:temporary (:sc double-avx2-reg) %ymm0)
    (:temporary (:sc double-avx2-reg) %ymm1)
    (:temporary (:sc double-avx2-reg) %ymm2)
    (:temporary (:sc double-avx2-reg) %ymm3)
    (:results (result :scs (double-avx2-reg)))
    (:result-types simd-pack-256-double)
    (:generator 8 ;; what should be the cost?
		(inst vcvtpd2ps %ymm0 x)
		(inst vrsqrtps %ymm0 %ymm0)
		(inst vcvtps2pd %ymm0 %ymm0)
		(inst vmulpd %ymm2 threehalfs %ymm0)
		(inst vmulpd %ymm3 %ymm0 %ymm0)
		(inst vmulpd %ymm1 x %ymm0)
		(inst vmulpd %ymm0 %ymm1 %ymm3)
		(inst vmulpd %ymm0 half %ymm0)
		(inst vsubpd %ymm0 %ymm2 %ymm0)
		(inst vmulpd %ymm2 threehalfs %ymm0)
		(inst vmulpd %ymm3 %ymm0 %ymm0)
		(inst vmulpd %ymm1 x %ymm0)
		(inst vmulpd %ymm0 %ymm1 %ymm3)
		(inst vmulpd %ymm0 half %ymm0)
		(inst vsubpd result %ymm2 %ymm0)))

  ;; (defknown (%d4dot) (%d4 %d4) %d2
  ;;     (movable flushable always-translatable)
  ;;   :overwrite-fndb-silently t)
  ;; (define-vop (%d4dot)
  ;;   (:translate %d4dot)
  ;;   (:policy :fast-safe)
  ;;   (:args (x :scs (double-avx2-reg)))
  ;;   (:args (y :scs (double-avx2-reg)))
  ;;   (:arg-types simd-pack-256-double simd-pack-256-double)
  ;;   (:results (result :scs (double-sse-reg)))
  ;;   (:result-types simd-pack-double)
  ;;   (:generator 4 (inst vdppd result x y #b1111)))
  
  (defknown (%d4hsum) (%d4) %d2
      (movable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4hsum)
    (:translate %d4hsum)
    (:policy :fast-safe)
    (:args (x :scs (double-avx2-reg)))
    (:arg-types simd-pack-256-double)
    (:temporary (:sc double-sse-reg) %xmm0)
    (:temporary (:sc double-sse-reg) %xmm1)
    (:results (result :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:generator 4 ;; what should be the cost?
		(inst vmovapd %xmm0 x)
		(inst vextractf128 %xmm1 x 1)
		(inst vzeroupper)
		(inst vaddpd %xmm0 %xmm0 %xmm1)
		(inst vunpckhpd %xmm1 %xmm0 %xmm0)
		(inst vaddsd result %xmm0 %xmm1)))

  ;; (defknown (%s8hsum) (%s8) %s4
  ;;     (movable flushable always-translatable)
  ;;   :overwrite-fndb-silently t)
  ;; (define-vop (%s8hsum)
  ;;   (:translate %s8hsum)
  ;;   (:policy :fast-safe)
  ;;   (:args (x :scs (single-avx2-reg))
  ;;   (:arg-types simd-pack-256-single)
  ;;   (:temporary (:sc single-sse-reg) %xmm0)
  ;;   (:temporary (:sc single-sse-reg) %xmm1)
  ;;   (:results (result :scs (single-sse-reg)))
  ;;   (:result-types simd-pack-single)
  ;;   (:generator 4 ;; what should be the cost?
  ;; 		(inst vmovaps %xmm0 x)
  ;; 		(inst vextractf128 %xmm1 x 1)
  ;; 		(inst vzeroupper)
  ;; 		(inst vaddps %xmm0 %xmm0 %xmm1)
  ;; 		(inst vmovshdup xmm1 xmm0)
  ;; 		(inst vaddps xmm0 xmm0 xmm1)
  ;; 		(inst vmovhlps %xmm1 %xmm1 %xmm0)
  ;; 		(inst vaddss result %xmm0 %xmm1)))

 ;; vextractf128 xmm1,ymm0,0x1
 ;; vaddps xmm0,xmm0,xmm1
 ;; vmovshdup xmm1,xmm0
 ;; vaddps xmm0,xmm0,xmm1
 ;; vmovhlps xmm1,xmm1,xmm0
 ;; vaddss xmm0,xmm0,xmm1

  (defmacro macro-when (condition &body body)
    (when condition
      `(progn
	 ,@body)))

  ;; Process if AVX2/FMA3 supports is available
  (macro-when (member :avx2 sb-impl:+internal-features+)
    (defmacro loadu-pd (load-to-ymm input-vector index &optional (offset 0))
      `(inst vmovupd ,load-to-ymm
             (float-ref-ea ,input-vector ,index ,offset 8
			   :scale (ash 2 (- word-shift n-fixnum-tag-bits)))))
    (defmacro vfmuladd-pd (add-to-ymm mult-with-ymm
			   input-vector index &optional (offset 0))
      `(inst vfmadd231pd ,add-to-ymm ,mult-with-ymm
             (float-ref-ea ,input-vector ,index ,offset 8
			   :scale (ash 2 (- word-shift n-fixnum-tag-bits)))))
    (defknown %d4vdot-avx2 ((simple-array double-float (*))
			    (simple-array double-float (*))
			    fixnum)
	(simd-pack double-float)
	(movable flushable always-translatable)
      :overwrite-fndb-silently t)
    (define-vop (%d4vdot-avx2)
      (:translate %d4vdot-avx2)
      (:policy :fast-safe)
      (:args (u :scs (descriptor-reg))
	     (v :scs (descriptor-reg))
	     (n0-tn :scs (signed-reg)))
      (:arg-types simple-array-double-float simple-array-double-float
		  tagged-num)
      (:temporary (:sc signed-reg) i)
      (:temporary (:sc signed-reg) n0)
      (:temporary (:sc double-avx2-reg) ymm0)
      (:temporary (:sc double-avx2-reg) ymm1)
      (:temporary (:sc double-avx2-reg) ymm2)
      (:temporary (:sc double-avx2-reg) ymm3)
      (:temporary (:sc double-avx2-reg) ymm4)
      (:temporary (:sc double-avx2-reg) ymm5)
      (:temporary (:sc double-avx2-reg) ymm6)
      (:temporary (:sc double-avx2-reg) ymm7)
      (:temporary (:sc double-sse-reg) xmm8)
      (:temporary (:sc double-sse-reg) xmm9)
      (:results (result :scs (double-sse-reg)))
      (:result-types simd-pack-double)
      (:generator 25
		  (move n0 n0-tn) ;;initializing registers
		  (inst vxorpd ymm7 ymm7 ymm7) 
		  (inst vxorpd ymm6 ymm6 ymm6)
		  (inst vxorpd ymm5 ymm5 ymm5)
		  (inst vxorpd ymm4 ymm4 ymm4)
		  (inst xor i i)
		  LOOP ;; Start calculation loop
		  ;; Load 16 elements of U into temp registers
		  (loadu-pd ymm3 u i 12)
		  (loadu-pd ymm2 u i 8)
		  (loadu-pd ymm1 u i 4)
		  (loadu-pd ymm0 u i 0)
		  ;; Do 4 Fused Multiply-Adds with elements of U and V
		  (vfmuladd-pd ymm7 ymm3 v i 0)
		  (vfmuladd-pd ymm6 ymm2 v i 4)
		  (vfmuladd-pd ymm5 ymm1 v i 8)
		  (vfmuladd-pd ymm4 ymm0 v i 12)
		  (inst add i 16) ;; Index i by 16
		  (inst cmp i n0) ;; Do loop untill n0
		  (inst jmp :b LOOP) 
		  DONE
		  (inst vaddpd ymm4 ymm4 ymm5) ;; Summ up the 4 temp registers
		  (inst vaddpd ymm6 ymm6 ymm7)
		  (inst vaddpd ymm4 ymm4 ymm6)
		  (inst vmovapd xmm8 ymm4)        ;; Horizontal summ from here
		  (inst vextractf128 xmm9 ymm4 1)
		  (inst vzeroupper)
		  (inst vaddpd xmm8 xmm8 xmm9)
		  (inst vunpckhpd xmm9 xmm8 xmm8)
		  (inst vaddsd result xmm8 xmm9))))
  
  (defknown %vzeroupper () (integer)
      (always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%vzeroupper)
    (:translate %vzeroupper)
    (:policy :fast-safe)
    (:generator 1 (inst vzeroupper)))

  (defknown %d4zero () %d4+
      (movable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%d4zero)
    (:translate %d4zero)
    (:policy :fast-safe)
    (:results (result :scs (double-avx2-reg)))
    (:result-types simd-pack-256-double)
    (:generator 1 (inst vxorpd result result result)))

  (defknown %s8zero () %s8+
      (movable flushable always-translatable)
    :overwrite-fndb-silently t)
  (define-vop (%s8zero)
    (:translate %s8zero)
    (:policy :fast-safe)
    (:results (result :scs (single-avx2-reg)))
    (:result-types simd-pack-256-single)
    (:generator 1 (inst vxorps result result result)))

  (declaim (ftype (function (double-float double-float
			     double-float double-float)
			    (simd-pack-256 double-float)) %make-avx-double))
  (declaim (inline %make-avx-double))
  (defun %make-avx-double (a b c d)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type double-float a b c d))
    (sb-vm::%make-simd-pack-256-double a b c d))
  
  (declaim (ftype (function (single-float single-float
			     single-float single-float
			     single-float single-float
			     single-float single-float)
			    (simd-pack-256 single-float)) %make-avx-single))
  (declaim (inline %make-avx-single))
  (defun %make-avx-single (a b c d e f g h)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type single-float a b c d e f g h))
    (sb-vm::%make-simd-pack-256-single a b c d e f g h))
  )

(in-package #:cl-user)

(deftype %d4  () '(simd-pack-256 double-float))
(deftype %d4+ () '(simd-pack-256 (double-float 0.0d0)))
(deftype %s8  () '(simd-pack-256 single-float))
(deftype %s8+ () '(simd-pack-256 (single-float 0.0)))
(deftype %d2  () '(simd-pack double-float))

(defmacro define-constant (name value &optional doc)
  `(defconstant ,name (if (boundp ',name) (symbol-value ',name) ,value)
     ,@(when doc (list doc))))

(declaim (inline make-avx-double))
(defun make-avx-double (a &optional (b a) (c a) (d a))
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (cond ((typep a 'double-float) (sb-vm::%make-avx-double a b c d))
	((typep a 'single-float) (sb-vm::%make-avx-double
				  (float a 0d0) (float b 0d0)
				  (float c 0d0) (float d 0d0)))
	((typep a 'integer     ) (sb-vm::%make-avx-double
				  (float a 0d0) (float b 0d0)
				  (float c 0d0) (float d 0d0)))))

(declaim (inline make-avx-single))
(defun make-avx-single (a &optional (b a) (c a) (d a) (e a) (f a) (g a) (h a))
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (cond ((typep a 'single-float)
	   (sb-vm::%make-avx-single a b c d e f g h))
	((typep a 'double-float) (sb-vm::%make-avx-single
				  (float a 0s0) (float b 0s0) (float c 0s0)
				  (float d 0s0) (float e 0s0) (float f 0s0)
				  (float g 0s0) (float h 0s0)))
	((typep a 'integer     ) (sb-vm::%make-avx-single
				  (float a 0s0) (float b 0s0) (float c 0s0)
				  (float d 0s0) (float e 0s0) (float f 0s0)
				  (float g 0s0) (float h 0s0)))))

(defparameter %0.0d4 (make-avx-double 0.0d0))
(defparameter %0.0s8 (make-avx-single 0.0))
(defparameter %1.0d4 (make-avx-double 1.0d0))
(defparameter %1.0s8 (make-avx-single 1.0))
(defparameter %-1.0d4 (make-avx-double -1.0d0))
(define-constant %0.0d4c (make-avx-double 0.0d0))
(define-constant %0.0s8c (make-avx-single 0.0))
(define-constant %1.0d4c (make-avx-double 1.0d0))
(define-constant %1.0s8c (make-avx-single 1.0))
(define-constant %-1.0d4c (make-avx-double -1.0d0))
(define-constant %0.5d4c (make-avx-double 0.5d0))
(define-constant %1.5d4c (make-avx-double 1.5d0))
(define-constant %2.0d4c (make-avx-double 2.0d0))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (declaim (inline d4+))
  (declaim (ftype (function (&rest (simd-pack-256 double-float))
			    (simd-pack-256 double-float)) d4+))
  (defun d4+ (&rest args)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (cond ((null args) %0.0d4c)
        ((null (cdr args)) (car args))
        ((null (cddr args)) (sb-vm::%d4+ (car args) (cadr args)))
        (t (sb-vm::%d4+ (car args) (apply #'d4+ (cdr args))))))
  (define-compiler-macro d4+ (&whole whole &rest args &environment env)
  (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
         (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
      (let ((args (case (car whole)
		    (apply (nconc (butlast (cdr args)) (car (last (cdr args)))))
                    (funcall args)
                    (t args))))
        (cond ((null args) %0.0d4c)
              ((null (cdr args)) (car args))
              ((null (cddr args)) `(sb-vm::%d4+ ,(car args) ,(cadr args)))
              (t `(sb-vm::%d4+ ,(car args)
			       ,(funcall (compiler-macro-function 'd4+)
			       `(funcall #'d4+ ,@(cdr args)) env))))) whole))

  (declaim (inline s8+))
  (declaim (ftype (function (&rest (simd-pack-256 single-float))
			    (simd-pack-256 single-float)) s8+))
  (defun s8+ (&rest args)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (cond ((null args) %0.0s8c)
        ((null (cdr args)) (car args))
        ((null (cddr args)) (sb-vm::%s8+ (car args) (cadr args)))
        (t (sb-vm::%s8+ (car args) (apply #'s8+ (cdr args))))))
  (define-compiler-macro s8+ (&whole whole &rest args &environment env)
  (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
         (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
      (let ((args (case (car whole)
		    (apply (nconc (butlast (cdr args)) (car (last (cdr args)))))
                    (funcall args)
                    (t args))))
        (cond ((null args) %0.0s8c)
              ((null (cdr args)) (car args))
              ((null (cddr args)) `(sb-vm::%s8+ ,(car args) ,(cadr args)))
              (t `(sb-vm::%s8+ ,(car args)
			       ,(funcall (compiler-macro-function 's8+)
					 `(funcall #'s8+ ,@(cdr args)) env))))) whole))
  
  (declaim (inline d4-))
  (declaim (ftype (function (&rest (simd-pack-256 double-float))
			    (simd-pack-256 double-float)) d4-))
  (defun d4- (arg &rest args)
  (declare (optimize (speed 3)))
  (cond ((null args) (sb-vm::%d4- %0.0d4c arg))
        ((null (cdr args)) (sb-vm::%s8- arg (car args)))
        (t (apply #'s8- (sb-vm::%s8- arg (car args)) (cdr args)))))
  (define-compiler-macro d4- (&whole whole arg &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(cond ((null args) `(sb-vm::%d4- %0.0d4c ,arg))
              ((null (cdr args)) `(sb-vm::%d4- ,arg ,(car args)))
              (t (funcall (compiler-macro-function 'd4-)
			  `(funcall #'d4- (sb-vm::%d4- arg (car args))
				    ,@(cdr args)) env))) whole))

  (declaim (inline s8-))
  (declaim (ftype (function (&rest (simd-pack-256 single-float))
			    (simd-pack-256 single-float)) s8-))
  (defun s8- (arg &rest args)
  (declare (optimize (speed 3)))
  (cond ((null args) (sb-vm::%s8- %0.0s8c arg))
        ((null (cdr args)) (sb-vm::%s8- arg (car args)))
        (t (apply #'s8- (sb-vm::%s8- arg (car args)) (cdr args)))))
  (define-compiler-macro s8- (&whole whole arg &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(cond ((null args) `(sb-vm::%s8- %0.0s8c ,arg))
              ((null (cdr args)) `(sb-vm::%s8- ,arg ,(car args)))
              (t (funcall (compiler-macro-function 's8-)
			  `(funcall #'s8- (sb-vm::%s8- arg (car args))
				    ,@(cdr args)) env))) whole))

  (declaim (inline d4*))
  (declaim (ftype (function (&rest (simd-pack-256 double-float))
			    (simd-pack-256 double-float)) d4*))
  (defun d4* (&rest args)
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (cond ((null args) %1.0d4c)
          ((null (cdr args)) (car args))
          ((null (cddr args)) (sb-vm::%d4* (car args) (cadr args)))
          (t (sb-vm::%d4* (car args) (apply 'd4* (cdr args))))))
  (define-compiler-macro d4* (&whole whole &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(let ((args (case (car whole)
                      (apply (nconc (butlast (cdr args)) (car (last (cdr args)))))
                      (funcall args)
                      (t args))))
          (cond ((null args) %1.0d4c)
		((null (cdr args)) (car args))
		((null (cddr args)) `(sb-vm::%d4* ,(car args) ,(cadr args)))
		(t `(sb-vm::%d4* ,(car args)
				 ,(funcall (compiler-macro-function 'd4*)
					   `(funcall #'d4* ,@(cdr args)) env))))) whole))

  (declaim (inline s8*))
  (declaim (ftype (function (&rest (simd-pack-256 single-float))
			    (simd-pack-256 single-float)) s8*))
  (defun s8* (&rest args)
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (cond ((null args) %1.0s8c)
          ((null (cdr args)) (car args))
          ((null (cddr args)) (sb-vm::%s8* (car args) (cadr args)))
          (t (sb-vm::%s8* (car args) (apply 's8* (cdr args))))))
  (define-compiler-macro s8* (&whole whole &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(let ((args (case (car whole)
                      (apply (nconc (butlast (cdr args)) (car (last (cdr args)))))
                      (funcall args)
                      (t args))))
          (cond ((null args) %1.0s8c)
		((null (cdr args)) (car args))
		((null (cddr args)) `(sb-vm::%s8* ,(car args) ,(cadr args)))
		(t `(sb-vm::%s8* ,(car args)
				 ,(funcall (compiler-macro-function 's8*)
				 `(funcall #'s8* ,@(cdr args)) env))))) whole))

  (declaim (inline d4/))
  (declaim (ftype (function (&rest (simd-pack-256 double-float))
			    (simd-pack-256 double-float)) d4/))
  (defun d4/ (arg &rest args)
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (cond ((null args) (sb-vm::%d4/ %1.0d4c arg))
          ((null (cdr args)) (sb-vm::%d4/ arg (car args)))
          (t (apply #'d4/ (sb-vm::%d4/ arg (car args)) (cdr args)))))
  (define-compiler-macro d4/ (&whole whole arg &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(cond ((null args) `(sb-vm::%d4/ ,%1.0d4c ,arg))
              ((null (cdr args)) `(sb-vm::%d4/ ,arg ,(car args)))
              (t (funcall (compiler-macro-function 'd4/)
		   `(funcall #'d4/ (sb-vm::%d4/ ,arg ,(car args))
	       		     ,@(cdr args)) env))) whole))

  (declaim (inline s8/))
  (declaim (ftype (function (&rest (simd-pack-256 single-float))
			    (simd-pack-256 single-float)) s8/))
  (defun s8/ (arg &rest args)
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (cond ((null args) (sb-vm::%s8/ %1.0s8c arg))
          ((null (cdr args)) (sb-vm::%s8/ arg (car args)))
          (t (apply #'s8/ (sb-vm::%s8/ arg (car args)) (cdr args)))))
  (define-compiler-macro s8/ (&whole whole arg &rest args &environment env)
    (if (> (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'speed)
           (sb-c::policy-quality (slot-value env 'sb-c::%policy) 'space))
	(cond ((null args) `(sb-vm::%s8/ ,%1.0s8c ,arg))
              ((null (cdr args)) `(sb-vm::%s8/ ,arg ,(car args)))
              (t (funcall (compiler-macro-function 's8/)
		   `(funcall #'s8/ (sb-vm::%s8/ ,arg ,(car args))
	       		     ,@(cdr args)) env))) whole))
  
  (defmacro define-double-unary-operation (name avx2-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (declaim (inline ,name))
       (declaim (ftype (function (%d4) %d4) ,name))
       (defun ,name (x)
	 (declare (optimize (speed 3) (safety 0) (debug 0))
		  (type %d4 x))
	 (,avx2-operation x))))
  (define-double-unary-operation d4sqrt sb-vm::%d4sqrt)

  (defmacro define-d2s-conv-unary-operation (name avx2-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (declaim (inline ,name))
       (declaim (ftype (function (%d4) %s4) ,name))
       (defun ,name (x)
	 (declare (optimize (speed 3) (safety 0) (debug 0))
		  (type %d4 x))
	 (,avx2-operation x))))
  (define-d2s-conv-unary-operation d4pd2ps sb-vm::%d4pd2ps)

  (defmacro define-single-unary-operation (name avx2-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (declaim (inline ,name))
       (declaim (ftype (function (%s4) %s4) ,name))
       (defun ,name (x)
	 (declare (optimize (speed 3) (safety 0) (debug 0))
		  (type %s4 x))
	 (,avx2-operation x))))
  (define-single-unary-operation s4rcpps sb-vm::%s4rcpps)
  (define-single-unary-operation s4rsqrt sb-vm::%s4rsqrt)

  (defmacro define-s2d-conv-unary-operation (name avx2-operation)
    `(eval-when (:compile-toplevel :load-toplevel :execute)
       (declaim (inline ,name))
       (declaim (ftype (function (%s4) %d4) ,name))
       (defun ,name (x)
	 (declare (optimize (speed 3) (safety 0) (debug 0))
		  (type %s4 x))
	 (,avx2-operation x))))
  (define-s2d-conv-unary-operation s4ps2pd sb-vm::%s4ps2pd)


  (declaim (inline d4ref))
  (declaim (ftype (function ((simple-array double-float (*))
			     (integer 0 #.most-positive-fixnum))
			    %d4) d4ref))
  (defun d4ref (v i)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type (simple-array double-float (*)) v)
	     (type (integer 0 #.most-positive-fixnum) i))
    (sb-vm::%d4ref v i))

  (declaim (inline s8ref))
  (declaim (ftype (function ((simple-array single-float (*))
			     (integer 0 #.most-positive-fixnum))
			    %s8) s8ref))
  (defun s8ref (v i)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type (simple-array single-float (*)) v)
	     (type (integer 0 #.most-positive-fixnum) i))
    (sb-vm::%s8ref v i))
  
  (declaim (inline (setf d4ref)))
  (declaim (ftype (function (%d4 (simple-array double-float (*))
				 (integer 0 #.most-positive-fixnum))
			    %d4) (setf d4ref)))
  (defun (setf d4ref) (new-value v i)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type (simple-array double-float (*)) v)
	     (type (integer 0 #.most-positive-fixnum) i)
	     (type %d4 new-value))
    (sb-vm::%d4set v i new-value))

  (declaim (inline (setf s8ref)))
  (declaim (ftype (function (%s8 (simple-array single-float (*))
				 (integer 0 #.most-positive-fixnum))
			    %s8) (setf s8ref)))
  (defun (setf s8ref) (new-value v i)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type (simple-array single-float (*)) v)
	     (type (integer 0 #.most-positive-fixnum) i)
	     (type %s8 new-value))
    (sb-vm::%s8set v i new-value))
  
  (declaim (inline vzeroupper))
  (declaim (ftype (function () integer) vzeroupper))
  (defun vzeroupper ()
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (sb-vm::%vzeroupper))

  (declaim (inline d4zero))
  (declaim (ftype (function () %d4+) d4zero))
  (defun d4zero ()
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (sb-vm::%d4zero))

  (declaim (inline s8zero))
  (declaim (ftype (function () %s8+) s8zero))
  (defun s8zero ()
    (declare (optimize (speed 3) (safety 0) (debug 0)))
    (sb-vm::%s8zero))
  
  (declaim (inline d4rsqrt))
  (declaim (ftype (function (%d4+) %d4+) d4rsqrt))
  (defun d4rsqrt (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4+ %x))
    (sb-vm::%d4rsqrt %x %0.5d4c %1.5d4c))
  
  (declaim (inline d4rec))
  (declaim (ftype (function (%d4) %d4) d4rec))
  (defun d4rec (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (sb-vm::%d4rec %x %2.0d4c))
  
  (declaim (inline avx-doubles))
  (defun avx-doubles (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (multiple-value-list (sb-vm::%simd-pack-256-doubles %x)))
  
  (declaim (inline avx-double-0))
  (defun avx-double-0 (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (nth 0 (avx-doubles %x)))
  
  (declaim (inline avx-double-1))
  (defun avx-double-1 (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (nth 1 (avx-doubles %x)))
  
  (declaim (inline avx-double-2))
  (defun avx-double-2 (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (nth 2 (avx-doubles %x)))
  
  (declaim (inline avx-double-3))
  (defun avx-double-3 (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (nth 3 (avx-doubles %x)))

  (declaim (inline avx-singles))
  (defun avx-singles (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %s8 %x))
    (multiple-value-list (sb-vm::%simd-pack-256-singles %x)))
  
  (define-modify-macro d4incf (&optional (num %1.0d4c)) d4+)
  (define-modify-macro d4decf (&optional (num %1.0d4c)) d4-)
  (define-modify-macro s8incf (&optional (num %1.0s8c)) s8+)
  (define-modify-macro s8decf (&optional (num %1.0s8c)) s8-)
  
  (declaim (inline sse-doubles))
  (defun sse-doubles (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d2 %x))
    (multiple-value-list (sb-vm::%simd-pack-doubles %x)))
  
  (declaim (inline sse-double-low))
  (declaim (ftype (function (%d2) double-float) sse-double-low))
  (defun sse-double-low (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d2 %x))
    (sb-vm::%simd-pack-double-item %x 1))
  
  (declaim (inline sse-double-high))
  (declaim (ftype (function (%d2) double-float) sse-double-high))
  (defun sse-double-high (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d2 %x))
    (sb-vm::%simd-pack-double-item %x 0))

  (declaim (inline sse-single-high))
  (declaim (ftype (function (%s4) single-float) sse-single-high))
  (defun sse-single-high (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %s4 %x))
    (sb-vm::%simd-pack-single-item %x 0))
  
  (declaim (inline d4hsum))
  (declaim (ftype (function (%d4) double-float) d4hsum))
  (defun d4hsum (%x)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x))
    (sb-vm::%simd-pack-double-item (sb-vm::%d4hsum %x) 0))

  ;; (declaim (inline s8hsum))
  ;; (declaim (ftype (function (%s8) single-float) s8hsum))
  ;; (defun s8hsum (%x)
  ;;   (declare (optimize (speed 3) (safety 0) (debug 0))
  ;; 	     (type %s8 %x))
  ;;   (sb-vm::%simd-pack-single-item (sb-vm::%s8hsum %x) 0))
  
  (declaim (inline d4dot1))
  (declaim (ftype (function (%d4 %d4) double-float) d4dot1))
  (defun d4dot1 (%x %y)
    (declare (optimize (speed 3) (safety 0) (debug 0))
	     (type %d4 %x %y))
    (d4hsum (d4* %x %y)))

  ;; (declaim (inline d4dot))
  ;; (declaim (ftype (function (%d4 %d4) double-float) d4dot))
  ;; (defun d4dot (%x %y)
  ;;   (declare (optimize (speed 3) (safety 0) (debug 0))
  ;; 	     (type %d4 %x %y))
  ;;   (d4hsum (sb-vm::%d4dot %x %y (the (unsigned-byte 8) #b1111))))
  
  (defmacro macro-when (condition &body body)
    (when condition
      `(progn
	 ,@body)))
  
  (defun iota (n &key (start 0) (step 1))
    "iota using DO iteration macro"
    (when (minusp n)
      (error "Invalid number of items specified"))
    (do ((i 0 (1+ i))
	 (item start (+ item step))
	 (result nil (push item result)))
	((= i n) (nreverse result))))
  
  (defmacro with-gensyms (syms &body body)
    `(let ,(loop for s in syms collect `(,s (gensym)))
       ,@body))
  
  (defmacro time-average (n &body body)
    "N-average the execution time of BODY in seconds"
    (with-gensyms (start end)
      `(let (,start ,end)
	 (setq ,start (get-internal-real-time))
	 (loop for i below ,n
	       do ,@body)
	 (setq ,end (get-internal-real-time))
	 (/ (- ,end ,start) ,n internal-time-units-per-second))))
  
  (defmacro time-total (n &body body)
    "N-average the execution time of BODY in seconds"
    (with-gensyms (start end)
      `(let (,start ,end)
	 (setq ,start (get-internal-real-time))
	 (loop for i below ,n
	       do ,@body)
	 (setq ,end (get-internal-real-time))
	 (coerce (/ (- ,end ,start) internal-time-units-per-second)
		 'float))))

  (declaim (ftype (function ((unsigned-byte 64) (unsigned-byte 64)
			     &rest (unsigned-byte 64))
                            (unsigned-byte 64))
                  min-factor)
           (inline min-factor))
  (defun min-factor (x y0 &rest ys)
    "calculate x0*y with the minimum m where x = x0*y + m, y = y0*y1*y2*..., and ys = (y1, y2, ...)"
    (declare (type (unsigned-byte 64) x y0)
             (type list ys)
             (dynamic-extent ys))
    (let* ((y (reduce #'* ys :initial-value y0))
           (m (mod x y)))
      (declare (type (unsigned-byte 64) y m))
      (- x m)))
  
  ;; Process if AVX2+FMA3 supports is available
  (sb-vm::macro-when (member :avx2 sb-impl:+internal-features+)
    (declaim (notinline d4vdot-avx2))
    (declaim (ftype (function ((simple-array double-float (*))
			       (simple-array double-float (*)))
			      double-float) d4vdot-avx2))
    (defun d4vdot-avx2 (u v)
      (declare (optimize (speed 3) (safety 0) (debug 0) (space 0))
	       (type (simple-array double-float (*)) u v))
      (let* ((n  (min (length u) (length v)))
	     (n0 (- n (mod n 16)))) ;; 16 elements processed in 1 cycle 
	(declare (type fixnum n n0))
	(+ (sb-vm::%simd-pack-double-item (sb-vm::%d4vdot-avx2 u v n0) 0)
	   (if (> n 16) (loop for i of-type fixnum from n0 below n
			      summing (* (aref u i) (aref v i))
				into acu of-type double-float
			      finally (return acu)) 0.0d0)))))
  
  (declaim (inline d4vdot))
  (declaim (ftype (function ((simple-array double-float (*))
                             (simple-array double-float (*)))
                            double-float) d4vdot))
  (defun d4vdot (u v)
    (declare (optimize (speed 3) (safety 0) (debug 0) (space 0))
	     (type (simple-array double-float (*)) u v))
    (let* ((n  (min (length u) (length v)))
           (n0 (- n (mod n 4)))) 
      (declare (type fixnum n n0))
      (loop with %sum1 of-type %d4 = %0.0d4c
            for i of-type fixnum below n0 by 4
	    do (progn (d4incf %sum1 (d4* (d4ref u i) (d4ref v i))))
	    finally (return (+ (d4hsum %sum1)
			       (loop for i of-type fixnum from n0 below n
				     summing (* (aref u i) (aref v i))
				       into sum2 of-type double-float
				     finally (return sum2)))))))
  
  (declaim (inline vdot))
  (declaim (ftype (function ((simple-array double-float (*))
                             (simple-array double-float (*)))
                            double-float) vdot))
  (defun vdot (u v)
    (declare (optimize (speed 3) (safety 0) (debug 0) (space 0))
	     (type (simple-array double-float (*)) u v))
    (let* ((n  (min (length u) (length v)))
	   (n0 (- n (mod n 3))))
      (declare (type fixnum n n0))
      (loop with sum1 of-type double-float = 0.0d0
	    with sum2 of-type double-float = 0.0d0
	    with sum3 of-type double-float = 0.0d0 ;; Gets slower with sum4!
            for i of-type fixnum below n0 by 3
	    do (progn (incf sum1 (* (aref u i) (aref v i)))
		      (incf sum2 (* (aref u (1+ i)) (aref v (1+ i))))
		      (incf sum3 (* (aref u (+ 2 i)) (aref v (+ 2 i)))))
	    finally (return (+ sum1 sum2 sum3
			       (loop for i of-type fixnum from n0 below n
				     summing (* (aref u i) (aref v i))
				       into sum4 of-type double-float
				     finally (return sum4))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Benchmark

;; (defparameter u (make-array 5000 :element-type 'double-float
;;                    :initial-contents
;;                (mapcar (lambda (i) (+ i 0.5d0))
;; 		       (iota 5000 :start 0
;; 				 :step (random 0.5d0)))))

;; (defparameter v (make-array 5000 :element-type 'double-float
;;                                 :initial-contents
;;                                 (mapcar (lambda (i) (+ i 0.5d0))
;; 					(iota 5000 :start 0
;; 						   :step (random 1.5d0)))))

;; (defmacro repeat (times &body body)
;;     `(loop repeat ,times do
;;        ,@body))

;; (defun foo (u v)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (type (simple-array double-float (*)) u v))
;;   (d4vdot-avx2 u v))

;; (defun bar (u v)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (type (simple-array double-float (*)) u v))
;;   (d4vdot u v))

;; (defun baz (u v)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (type (simple-array double-float (*)) u v))
;;   (vdot u v))

;; (defun benchmark2-d4vdot-avx2 (n u v)
;;   (time (repeat n (d4vdot-avx2 u v))))

;; (defun benchmark-d4vdot-avx2 (repeat &rest v-lens)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (notinline d4vdot-avx2))
;;   (loop for len of-type fixnum in v-lens
;;      do (format t "Doing ~A ~%" len)
;;      collect (let ((u (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (* (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0)))))
;;                    (v (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (* (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0))))))
;; 	       (declare (type (simple-array double-float (*)) u v))
;; 	       (time-total repeat (d4vdot-avx2 u v))
;; 	       ;(time-total repeat (bar u v))
;; 	       ;(time-total repeat (baz u v))
;; 	       )))

;; (defun benchmark-d4vdot (repeat &rest v-lens)
;;   (declare (optimize speed (safety 0) (debug 0)))
;;   (loop for len of-type fixnum in v-lens
;;      do (format t "Doing ~A ~%" len)
;;      collect (let ((u (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (* (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0)))))
;;                    (v (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (* (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0))))))
;; 	       (declare (type (simple-array double-float (*)) u v))
;; 	       ;(time-total repeat (foo u v))
;; 	       (time-total repeat (bar u v))
;; 	       ;(time-total repeat (baz u v))
;; 	       )))

;; (defun benchmark-vdot (repeat &rest v-lens)
;;   (declare (optimize speed (safety 0) (debug 0)))
;;   (loop for len of-type fixnum in v-lens
;;      do (format t "Doing ~A ~%" len)
;;      collect (let ((u (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (+ (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0)))))
;;                    (v (make-array len
;;                                   :element-type 'double-float
;;                                   :initial-contents
;;                                   (mapcar (lambda (i) (+ (+ i 1) (get-internal-real-time) 0.5d0))
;; 					  (iota len :start 0
;; 						    :step (random 1000.5d0))))))
;; 	       (declare (type (simple-array double-float (*)) u v))
;; 	       ;(time-total repeat (foo u v))
;; 	       ;(time-total repeat (bar u v))
;; 	       (time-total repeat (baz u v))
;; 	       )))

;; (defun benchmark-d4vdot (u v &rest repeats)
;;   (declare (type (simple-array double-float *) u v))
;;   (loop for repeat of-type fixnum in repeats
;; 	do (format t "Doing ~A ~%" repeat)
;; 	collect (time-total repeat (d4vdot-avx2 u v))))

;; (defun benchmark-d4vdot-avx2 (u v &rest repeats)
;;   (declare (optimize (speed 0))
;; 	   (type (simple-array double-float (*)) u v))
;;   (loop for repeat in repeats
;;      do (format t "Doing ~A ~%" repeat)
;;      collect (time-total repeat (d4vdot-avx2 u v))))

;(benchmark 1e6 100 500 1000 5000 50000 500000)

;; (defun benchmark-d4vdot (n m)
;;   (let* ((u  (make-array n :element-type 'double-float
;; 			   :initial-contents
;;                            (mapcar (lambda (i) (+ i 0.5d0))
;; 				   (iota n :start 0
;; 					   :step (random 0.5d0)))))
;; 	 (v  (make-array n :element-type 'double-float
;; 			   :initial-contents
;;                            (mapcar (lambda (i) (+ i 0.5d0))
;; 				   (iota n :start 0
;; 					   :step (random 1.5d0))))))
;;     (declare (type (simple-array double-float (*)) u v))
;;     (time (loop repeat m do (d4vdot-avx2 u v)))
;;     (print (d4vdot-avx2 u v))
;;     (time (loop repeat m do (d4vdot u v)))
;;     (print (d4vdot u v))
;;     ;; (time (loop repeat m do (vdot u v)))
;;     ;; (print (vdot u v))
;;     ))

;; Array lengths of 5000 and 1 million iterationa
;(benchmark-d4vdot 5000 1000000)

;; (let* ((n 50000)
;;        (u  (make-array n :element-type 'double-float
;; 			  :initial-contents
;;                           (mapcar (lambda (i) (+ i 0.5d0))
;; 				  (iota n :start 0
;; 					  :step (random 0.5d0)))))
;;        (v  (make-array n :element-type 'double-float
;; 			 :initial-contents
;;                          (mapcar (lambda (i) (+ i 0.5d0))
;; 				 (iota n :start 0
;; 					 :step (random 1.5d0))))))
;;   (declare (type (simple-array double-float (*)) u v))
;;   (print (d4vdot u v))
;;   (print (d4vdot-avx2 u v))
;;   nil)

;; (ql:quickload "cl3a" :silent t)
;; (defun run-bench-cl3a (n m)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (type fixnum n m)
;; 	   (notinline cl3a:dv*v))
;;   (let ((va (cl3a:make-vec-init n 'double-float))
;;         (vb (cl3a:make-vec-init n 'double-float)))
;;     (declare (type (cl3a:vec double-float) va vb))
;;     (dotimes (i n)
;;       (setf (cl3a:vecref va i) (random 1d0))
;;       (setf (cl3a:vecref vb i) (random 1d0)))
;;     (time (loop :repeat m :do (cl3a:dv*v va vb)))))

;; (defun run-bench-d4 (n m)
;;   (declare (optimize speed (safety 0) (debug 0))
;; 	   (type fixnum n m)
;; 	   (notinline d4vdot-avx2))
;;   (let ((va (make-array n :element-type 'double-float))
;;         (vb (make-array n :element-type 'double-float)))
;;     (declare (type (simple-array double-float *) va vb))
;;     (dotimes (i n)
;;       (setf (aref va i) (random 1d0))
;;       (setf (aref vb i) (random 1d0)))
;;     (time (loop repeat m do (d4vdot-avx2 va vb)))))
