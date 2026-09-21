fn target_is_macos() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
}

#[cfg(feature = "cpp")]
fn main() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .flag("-std=c++11")
        .static_crt(true)
        .file("src/esaxx.cpp")
        .include("src");
    if target_is_macos() {
        build.flag("-stdlib=libc++");
    }
    build.compile("esaxx");
}

#[cfg(not(feature = "cpp"))]
fn main() {}