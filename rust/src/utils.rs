use std::ffi::CString;
use std::os::raw::c_char;
use winapi::um::libloaderapi::LoadLibraryA;

pub fn load_cuda_dll() {
    // There is a bug with tch-rs at 0.18.0 and above:
    // torch_cuda.dll isn't included in this package, making it unaccesible to CUDA.
    // So it's necessary to mannually import torch_cuda.dll.
    let path = CString::new("F:\\libtorch_251_cu124\\lib\\torch_cuda.dll").unwrap();
    unsafe {
        LoadLibraryA(path.as_ptr() as *const c_char);
    }
}

pub const DROPLET_CLASS_LABELS: [&str; 4] = [
    "0", "1", "2", "3"
];