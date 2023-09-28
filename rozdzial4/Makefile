all: gpu cpu

gpu:
	pgcc -acc -mp -O3 -ta=tesla  alg1_b-c.c -o alg1_b-c_gpu
	pgcc -acc -mp -O3 -ta=tesla  alg1_c-w.c -o alg1_c-w_gpu
	pgcc -acc -mp -O3 -ta=tesla  alg1_r-w.c -o alg1_r-w_gpu
	pgcc -acc -mp -O3 -ta=tesla  alg1_r-w-c.c -o alg1_r-w-c_gpu
	pgcc -acc -mp -O3 -ta=tesla  alg2_b-c.c -o alg2_b-c_gpu
	pgcc -acc -mp -O3 -ta=tesla  alg2_c-w.c -o alg2_c-w_gpu

cpu:
	pgcc -acc -mp -O3 -ta=multicore  alg1_b-c.c -o alg1_b-c_cpu
	pgcc -acc -mp -O3 -ta=multicore  alg1_c-w.c -o alg1_c-w_cpu
	pgcc -acc -mp -O3 -ta=multicore  alg1_r-w.c -o alg1_r-w_cpu
	pgcc -acc -mp -O3 -ta=multicore  alg1_r-w-c.c -o alg1_r-w-c_cpu
	pgcc -acc -mp -O3 -ta=multicore  alg2_b-c.c -o alg2_b-c_cpu
	pgcc -acc -mp -O3 -ta=multicore  alg2_c-w.c -o alg2_c-w_cpu
