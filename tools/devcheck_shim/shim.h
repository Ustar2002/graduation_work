/* tools/devcheck_shim/shim.h -- devcheck 전용 libc 보정 헤더 (설계 v3 §14).
 *
 * gcc 가 없는 개발 머신(Windows)에서 tools/devcheck.py 가 `python -m ziglang cc -include shim.h` 로 C 커널을 빌드할 때
 * 모든 번역 단위 맨 앞에 넣는다. 측정 경로(harness/build/c.py 의 gcc 빌드)에서는 쓰지 않는다.
 *
 * Windows(UCRT)에는 C11 `aligned_alloc` 과 POSIX `posix_memalign` 이 없고, 정렬 할당(`_aligned_malloc`)은 `free` 가 아니라
 * `_aligned_free` 로 풀어야 한다. Linux(glibc)에서는 `aligned_alloc`·`posix_memalign`·`malloc` 을 모두 `free` 하나로 푼다.
 * 그 계약을 흉내 내려고 할당 함수 전체를 `_aligned_malloc` 계열 위에 다시 세운다:
 *
 *   malloc/calloc/realloc      -> _aligned_malloc/_aligned_realloc (16 바이트 정렬; glibc x86-64 의 malloc 정렬과 같다)
 *   aligned_alloc/memalign     -> _aligned_malloc (요청한 정렬; 2의 거듭제곱이 아니면 NULL + EINVAL)
 *   posix_memalign             -> 위와 같음 (정렬이 sizeof(void*) 의 2의 거듭제곱 배수가 아니면 EINVAL)
 *   _mm_malloc/_mm_free        -> 같은 계열 (Linux 에서는 free 와 섞어 써도 동작하므로 여기서도 그렇게 맞춘다)
 *   free                       -> _aligned_free
 *
 * 이름은 객체형 매크로로 바꾼다(호출과 함수 포인터 모두 잡힌다). 표준 헤더를 여기서 먼저 포함해 두므로 커널이 나중에
 * <stdlib.h> 를 포함해도 다시 선언되지 않는다. Windows 가 아니면 아무것도 바꾸지 않는다.
 *
 * 한계: `long` 은 Windows 에서 32비트다(LLP64). POSIX 전용 API(mmap 등)는 여기서 빌드되지 않는다. 논리 검증만 한다.
 */
#ifndef DEVCHECK_SHIM_H
#define DEVCHECK_SHIM_H

#if defined(_WIN32)

#include <errno.h>
#include <malloc.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <mm_malloc.h> /* _mm_malloc/_mm_free 의 원래 정의를 먼저 들여와 아래 매크로가 재정의하지 않게 한다 */
#endif

#define DEVCHECK_MALLOC_ALIGN ((size_t)16)

static inline void *devcheck_aligned_alloc(size_t alignment, size_t size) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        errno = EINVAL;
        return NULL;
    }
    return _aligned_malloc(size ? size : 1, alignment < sizeof(void *) ? sizeof(void *) : alignment);
}

static inline void *devcheck_malloc(size_t size) {
    return _aligned_malloc(size ? size : 1, DEVCHECK_MALLOC_ALIGN);
}

static inline void *devcheck_calloc(size_t count, size_t size) {
    size_t total;
    void *ptr;
    if (size != 0 && count > SIZE_MAX / size) {
        errno = ENOMEM;
        return NULL;
    }
    total = count * size;
    ptr = _aligned_malloc(total ? total : 1, DEVCHECK_MALLOC_ALIGN);
    if (ptr != NULL) memset(ptr, 0, total);
    return ptr;
}

static inline void *devcheck_realloc(void *ptr, size_t size) {
    if (ptr == NULL) return devcheck_malloc(size);
    if (size == 0) {
        _aligned_free(ptr);
        return NULL;
    }
    return _aligned_realloc(ptr, size, DEVCHECK_MALLOC_ALIGN);
}

static inline void devcheck_free(void *ptr) {
    _aligned_free(ptr); /* NULL 이면 아무 일도 하지 않는다 */
}

static inline int devcheck_posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *ptr;
    if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0) return EINVAL;
    ptr = _aligned_malloc(size ? size : 1, alignment);
    if (ptr == NULL) return ENOMEM;
    *memptr = ptr;
    return 0;
}

static inline void *devcheck_memalign(size_t alignment, size_t size) {
    return devcheck_aligned_alloc(alignment, size);
}

static inline void *devcheck_mm_malloc(size_t size, size_t alignment) {
    return devcheck_aligned_alloc(alignment, size);
}

static inline void devcheck_mm_free(void *ptr) {
    _aligned_free(ptr);
}

static inline char *devcheck_strdup(const char *text) {
    size_t n = strlen(text) + 1;
    char *copy = (char *)devcheck_malloc(n);
    if (copy != NULL) memcpy(copy, text, n);
    return copy;
}

#define malloc devcheck_malloc
#define calloc devcheck_calloc
#define realloc devcheck_realloc
#define free devcheck_free
#define aligned_alloc devcheck_aligned_alloc
#define posix_memalign devcheck_posix_memalign
#define memalign devcheck_memalign
#define _mm_malloc devcheck_mm_malloc
#define _mm_free devcheck_mm_free
#define strdup devcheck_strdup

#endif /* _WIN32 */

#endif /* DEVCHECK_SHIM_H */
