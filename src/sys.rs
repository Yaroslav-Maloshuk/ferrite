//! Portable process-memory telemetry.
//!
//! [`peak_rss_kb`] reports the calling process's resident-set high-water mark;
//! [`process_peak_rss_kb`] reports another process's. Linux (`VmHWM`) exposes
//! a true peak. macOS has no per-process peak API, so an *external* process is
//! sampled for its current RSS via `ps` — a lower bound — while the calling
//! process uses `getrusage` (`ru_maxrss`, a peak).

/// Linux `VmHWM` for `pid`, in KiB.
#[cfg(target_os = "linux")]
fn linux_vmhwm_kb(pid: u32) -> Option<u64> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

/// macOS current RSS for `pid` via `ps`, in KiB (not a peak).
#[cfg(target_os = "macos")]
fn macos_ps_rss_kb(pid: u32) -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// `getrusage(RUSAGE_SELF).ru_maxrss`, normalized to KiB.
#[cfg(unix)]
fn unix_self_maxrss_kb() -> u64 {
    let mut rusage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut rusage) } != 0 {
        return 0;
    }
    #[cfg(target_os = "macos")]
    {
        (rusage.ru_maxrss / 1024) as u64 // macOS reports bytes
    }
    #[cfg(not(target_os = "macos"))]
    {
        rusage.ru_maxrss as u64 // Linux/BSD report KiB
    }
}

/// Peak resident set of the calling process, in KiB.
pub fn peak_rss_kb() -> u64 {
    #[cfg(target_os = "linux")]
    let kb = linux_vmhwm_kb(std::process::id()).unwrap_or_else(unix_self_maxrss_kb);
    #[cfg(all(unix, not(target_os = "linux")))]
    let kb = unix_self_maxrss_kb();
    #[cfg(not(unix))]
    let kb = 0;
    kb
}

/// Peak resident set of another process, in KiB. Returns 0 if the process is
/// gone or the platform has no way to read it.
pub fn process_peak_rss_kb(pid: u32) -> u64 {
    #[cfg(target_os = "linux")]
    let kb = linux_vmhwm_kb(pid).unwrap_or(0);
    #[cfg(target_os = "macos")]
    let kb = macos_ps_rss_kb(pid).unwrap_or(0);
    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    let kb = 0;
    #[cfg(not(unix))]
    let kb = 0;
    kb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_peak_rss_is_positive() {
        assert!(peak_rss_kb() > 0);
    }

    #[test]
    fn own_pid_matches_self() {
        assert!(process_peak_rss_kb(std::process::id()) > 0);
    }

    #[test]
    fn unknown_pid_is_zero() {
        assert_eq!(process_peak_rss_kb(u32::MAX), 0);
    }
}
