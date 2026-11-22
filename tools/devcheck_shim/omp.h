/* tools/devcheck_shim/omp.h -- devcheck 전용 OpenMP 직렬 스텁 (설계 v3 §14).
 *
 * gcc 가 없는 개발 머신에서 tools/devcheck.py 가 `python -m ziglang cc` 로 C 커널을 빌드할 때만 쓴다
 * (`-I tools/devcheck_shim`, `-fopenmp` 없음). 측정 경로(harness/build/c.py 의 gcc 빌드)에서는 쓰지 않는다.
 *
 * - `#pragma omp ...` 는 컴파일러가 무시한다 → 병렬 영역은 스레드 하나로 순서대로 실행된다.
 * - 런타임 함수는 "스레드 1개인 팀"의 값을 돌려주는 static inline 이다(런타임 라이브러리 없음).
 * - `_OPENMP` 는 정의하지 않는다(`#ifdef _OPENMP` 로 감싼 코드는 직렬 분기를 탄다).
 * - `DEVCHECK_OMP_THREADS`(devcheck 가 환경변수에서 받아 `-D` 로 준다; 기본 1)는 `omp_get_max_threads()` 등 "쓸 수 있는
 *   스레드 수"가 돌려줄 값이다. 실행은 여전히 직렬이고 `omp_get_num_threads()` 는 1 이다. 조각 수를
 *   `omp_get_max_threads()` 로 정하는 커널의 여러 조각 경로를 순서대로 돌려 보려는 것이다.
 *
 * 즉 논리 검증만 한다: 경쟁 상태·리덕션 누락 같은 병렬 버그는 여기서 드러나지 않는다(Linux 에서 확인).
 */
#ifndef DEVCHECK_OMP_STUB_H
#define DEVCHECK_OMP_STUB_H

#include <time.h>

#ifndef DEVCHECK_OMP_THREADS
#define DEVCHECK_OMP_THREADS 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct omp_lock_t { int devcheck_locked; } omp_lock_t;
typedef struct omp_nest_lock_t { int devcheck_depth; } omp_nest_lock_t;

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
static inline void omp_set_num_threads(int num_threads) { (void)num_threads; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return DEVCHECK_OMP_THREADS; }
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_procs(void) { return DEVCHECK_OMP_THREADS; }
static inline int omp_get_thread_limit(void) { return DEVCHECK_OMP_THREADS; }
static inline int omp_in_parallel(void) { return 0; }
static inline int omp_in_final(void) { return 1; }

/* 동적 조정·중첩 */
static inline void omp_set_dynamic(int dynamic_threads) { (void)dynamic_threads; }
static inline int omp_get_dynamic(void) { return 0; }
static inline void omp_set_nested(int nested) { (void)nested; }
static inline int omp_get_nested(void) { return 0; }
static inline void omp_set_max_active_levels(int max_levels) { (void)max_levels; }
static inline int omp_get_max_active_levels(void) { return 1; }
static inline int omp_get_level(void) { return 0; }
static inline int omp_get_active_level(void) { return 0; }
static inline int omp_get_ancestor_thread_num(int level) { return level == 0 ? 0 : -1; }
static inline int omp_get_team_size(int level) { return level == 0 ? 1 : -1; }

/* 스케줄·배치 */
static inline void omp_set_schedule(omp_sched_t kind, int chunk_size) { (void)kind; (void)chunk_size; }
static inline void omp_get_schedule(omp_sched_t *kind, int *chunk_size) { *kind = omp_sched_static; *chunk_size = 0; }
static inline omp_proc_bind_t omp_get_proc_bind(void) { return omp_proc_bind_false; }
static inline int omp_get_num_places(void) { return 0; }
static inline int omp_get_place_num(void) { return -1; }
static inline int omp_get_cancellation(void) { return 0; }
static inline int omp_get_max_task_priority(void) { return 0; }

/* 잠금: 스레드가 하나라 상태만 기록한다 */
static inline void omp_init_lock(omp_lock_t *lock) { lock->devcheck_locked = 0; }
static inline void omp_destroy_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_set_lock(omp_lock_t *lock) { lock->devcheck_locked = 1; }
static inline void omp_unset_lock(omp_lock_t *lock) { lock->devcheck_locked = 0; }
static inline int omp_test_lock(omp_lock_t *lock) {
    if (lock->devcheck_locked) return 0;
    lock->devcheck_locked = 1;
    return 1;
}
static inline void omp_init_nest_lock(omp_nest_lock_t *lock) { lock->devcheck_depth = 0; }
static inline void omp_destroy_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_set_nest_lock(omp_nest_lock_t *lock) { lock->devcheck_depth++; }
static inline void omp_unset_nest_lock(omp_nest_lock_t *lock) { lock->devcheck_depth--; }
static inline int omp_test_nest_lock(omp_nest_lock_t *lock) { return ++lock->devcheck_depth; }

/* 시간 */
static inline double omp_get_wtime(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }
static inline double omp_get_wtick(void) { return 1.0 / (double)CLOCKS_PER_SEC; }

#ifdef __cplusplus
}
#endif

#endif /* DEVCHECK_OMP_STUB_H */
