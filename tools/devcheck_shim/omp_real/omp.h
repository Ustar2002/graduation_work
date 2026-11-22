/* tools/devcheck_shim/omp_real/omp.h -- devcheck 의 `--cc zig-omp` 전용 OpenMP 런타임 선언.
 *
 * gcc 가 없는 개발 머신(Windows)에서 c_openmp 커널을 **실제 스레드**로 돌려 보는 선택 기능이다:
 * `python -m ziglang cc -fopenmp -c` 로 clang 의 OpenMP 프런트엔드가 pragma 를 검사·코드 생성하게 하고, 목적 파일을
 * 이 머신의 LLVM OpenMP 런타임(Visual C++ 재배포 패키지의 `libomp140.x86_64.dll`)에 링크한다. zig 는 `omp.h` 를
 * 싣고 있지 않으므로 런타임이 내보내는 함수의 선언만 여기 둔다(직렬 스텁 `../omp.h` 와 달리 정의가 없다).
 *
 * 측정 경로(harness/build/c.py 의 gcc + libgomp)에서는 쓰지 않는다. 컴파일러도 런타임도 gcc/libgomp 가 아니므로
 * 여기서 통과한 것이 Linux 확인을 대신하지는 않는다 — 경쟁 상태·잘못된 pragma 를 일찍 드러내려는 것이다.
 * `_OPENMP` 는 `-fopenmp` 를 받은 컴파일러가 정의한다.
 */
#ifndef DEVCHECK_OMP_REAL_H
#define DEVCHECK_OMP_REAL_H

#ifdef __cplusplus
extern "C" {
#endif

/* LLVM libomp 의 잠금 타입(포인터 하나 크기의 불투명 값) */
typedef struct omp_lock_t { void *_lk; } omp_lock_t;
typedef struct omp_nest_lock_t { void *_lk; } omp_nest_lock_t;

typedef enum omp_sched_t {
    omp_sched_static = 1,
    omp_sched_dynamic = 2,
    omp_sched_guided = 3,
    omp_sched_auto = 4
} omp_sched_t;

typedef enum omp_proc_bind_t {
    omp_proc_bind_false = 0,
    omp_proc_bind_true = 1,
    omp_proc_bind_master = 2,
    omp_proc_bind_primary = 2,
    omp_proc_bind_close = 3,
    omp_proc_bind_spread = 4
} omp_proc_bind_t;

/* 스레드 수·팀 */
extern void omp_set_num_threads(int num_threads);
extern int omp_get_num_threads(void);
extern int omp_get_max_threads(void);
extern int omp_get_thread_num(void);
extern int omp_get_num_procs(void);
extern int omp_get_thread_limit(void);
extern int omp_in_parallel(void);
extern int omp_in_final(void);

/* 동적 조정·중첩 */
extern void omp_set_dynamic(int dynamic_threads);
extern int omp_get_dynamic(void);
extern void omp_set_nested(int nested);
extern int omp_get_nested(void);
extern void omp_set_max_active_levels(int max_levels);
extern int omp_get_max_active_levels(void);
extern int omp_get_level(void);
extern int omp_get_active_level(void);
extern int omp_get_ancestor_thread_num(int level);
extern int omp_get_team_size(int level);

/* 스케줄·배치 */
extern void omp_set_schedule(omp_sched_t kind, int chunk_size);
extern void omp_get_schedule(omp_sched_t *kind, int *chunk_size);
extern omp_proc_bind_t omp_get_proc_bind(void);
extern int omp_get_num_places(void);
extern int omp_get_place_num(void);
extern int omp_get_cancellation(void);
extern int omp_get_max_task_priority(void);

/* 잠금 */
extern void omp_init_lock(omp_lock_t *lock);
extern void omp_destroy_lock(omp_lock_t *lock);
extern void omp_set_lock(omp_lock_t *lock);
extern void omp_unset_lock(omp_lock_t *lock);
extern int omp_test_lock(omp_lock_t *lock);
extern void omp_init_nest_lock(omp_nest_lock_t *lock);
extern void omp_destroy_nest_lock(omp_nest_lock_t *lock);
extern void omp_set_nest_lock(omp_nest_lock_t *lock);
extern void omp_unset_nest_lock(omp_nest_lock_t *lock);
extern int omp_test_nest_lock(omp_nest_lock_t *lock);

/* 시간 */
extern double omp_get_wtime(void);
extern double omp_get_wtick(void);

#ifdef __cplusplus
}
#endif

#endif /* DEVCHECK_OMP_REAL_H */
