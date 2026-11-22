// 자리표시 파일: cargo 가 이 매니페스트를 읽으려면(`cargo fetch --locked --manifest-path …`) 라이브러리 소스가 있어야 한다.
// 하네스는 생성된 커널을 샘플 디렉터리의 `src/lib.rs` 에 쓴다. 이 파일은 샘플로 복사되지 않고, run 시작 때
// `harness.build.rust.probe_toolchain` 이 툴체인 점검용으로 빌드한다(알려진 정상 코드여야 한다: 비워 둔다).
