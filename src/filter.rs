use std::{
    ffi::c_void,
    mem::{size_of, transmute},
    sync::Arc,
};

use crate::{device::Device, sys::*, FilterError};

type HANDLE = isize;

/// A generic ray tracing denoising filter for denoising
/// images produces with Monte Carlo ray tracing methods
/// such as path tracing.
pub struct RayTracing<'b> {
    handle: OIDNFilter,
    device: Arc<Device>,
    albedo: Option<&'b [f32]>,
    normal: Option<&'b [f32]>,

    color_dx12: Option<(usize, HANDLE)>,
    albedo_dx12: Option<(usize, HANDLE)>,
    normal_dx12: Option<(usize, HANDLE)>,
    output_dx12: Option<(usize, HANDLE)>,

    hdr: bool,
    input_scale: f32,
    srgb: bool,
    clean_aux: bool,
    img_dims: (usize, usize),
}

pub unsafe extern "C" fn error(
    userPtr: *mut ::std::os::raw::c_void,
    code: OIDNError,
    message: *const ::std::os::raw::c_char,
) {
    println!("[{:?}] blah {:?}", code, std::ffi::CStr::from_ptr(message));
}

impl<'b> RayTracing<'b> {
    pub fn new(device: Arc<Device>) -> RayTracing<'b> {
        unsafe {
            oidnSetDeviceErrorFunction(device.0, Some(error), std::ptr::null_mut() as _);
        }
        unsafe {
            oidnRetainDevice(device.0);
        }
        let filter = unsafe { oidnNewFilter(device.0, b"RT\0" as *const _ as _) };
        RayTracing {
            handle: filter,
            device,
            albedo: None,
            normal: None,
            color_dx12: None,
            albedo_dx12: None,
            normal_dx12: None,
            output_dx12: None,
            hdr: false,
            input_scale: std::f32::NAN,
            srgb: false,
            clean_aux: false,
            img_dims: (0, 0),
        }
    }

    /// Set input auxiliary images containing the albedo and normals.
    ///
    /// Albedo must have three channels per pixel with values in [0, 1].
    /// Normal must contain the shading normal as three channels per pixel
    /// *world-space* or *view-space* vectors with arbitrary length, values
    /// in `[-1, 1]`.
    pub fn albedo_normal(&mut self, albedo: &'b [f32], normal: &'b [f32]) -> &mut RayTracing<'b> {
        self.albedo = Some(albedo);
        self.normal = Some(normal);
        self
    }

    /// Set an input auxiliary image containing the albedo per pixel (three
    /// channels, values in `[0, 1]`).
    pub fn albedo(&mut self, albedo: &'b [f32]) -> &mut RayTracing<'b> {
        self.albedo = Some(albedo);
        self
    }

    pub fn color_dx12(&mut self, size: usize, ptr: HANDLE) -> &mut RayTracing<'b> {
        self.color_dx12 = Some((size, ptr));
        self
    }

    pub fn albedo_dx12(&mut self, size: usize, ptr: HANDLE) -> &mut RayTracing<'b> {
        self.albedo_dx12 = Some((size, ptr));
        self
    }

    pub fn normal_dx12(&mut self, size: usize, ptr: HANDLE) -> &mut RayTracing<'b> {
        self.normal_dx12 = Some((size, ptr));
        self
    }

    pub fn output_dx12(&mut self, size: usize, ptr: HANDLE) -> &mut RayTracing<'b> {
        self.output_dx12 = Some((size, ptr));
        self
    }

    /// Set whether the color is HDR.
    pub fn hdr(&mut self, hdr: bool) -> &mut RayTracing<'b> {
        self.hdr = hdr;
        self
    }

    #[deprecated(since = "1.3.1", note = "Please use RayTracing::input_scale instead")]
    pub fn hdr_scale(&mut self, hdr_scale: f32) -> &mut RayTracing<'b> {
        self.input_scale = hdr_scale;
        self
    }

    /// Sets a scale to apply to input values before filtering, without scaling
    /// the output too.
    ///
    /// This can be used to map color or auxiliary feature values to the
    /// expected range. E.g. for mapping HDR values to physical units (which
    /// affects the quality of the output but not the range of the output
    /// values). If not set, the scale is computed implicitly for HDR images
    /// or set to 1 otherwise
    pub fn input_scale(&mut self, input_scale: f32) -> &mut RayTracing<'b> {
        self.input_scale = input_scale;
        self
    }

    /// Set whether the color is encoded with the sRGB (or 2.2 gamma) curve (LDR
    /// only) or is linear.
    ///
    /// The output will be encoded with the same curve.
    pub fn srgb(&mut self, srgb: bool) -> &mut RayTracing<'b> {
        self.srgb = srgb;
        self
    }

    /// Set whether the auxiliary feature (albedo, normal) images are
    /// noise-free.
    ///
    /// Recommended for highest quality but should not be enabled for noisy
    /// auxiliary images to avoid residual noise.
    pub fn clean_aux(&mut self, clean_aux: bool) -> &mut RayTracing<'b> {
        self.clean_aux = clean_aux;
        self
    }

    pub fn image_dimensions(&mut self, width: usize, height: usize) -> &mut RayTracing<'b> {
        self.img_dims = (width, height);
        self
    }

    pub fn filter(&self, color: &[f32], output: &mut [f32]) -> Result<(), FilterError> {
        self.execute_filter(Some(color), output)
    }

    pub fn filter_in_place(&self, color: &mut [f32]) -> Result<(), FilterError> {
        self.execute_filter(None, color)
    }

    pub fn setup(&self) -> Result<(), FilterError> {
        let pixelstride = size_of::<f32>() * 4;
        let bytes_per_row = 0;

        let color_ptr = unsafe {
            oidnNewSharedBufferFromWin32Handle(
                self.device.0,
                OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                std::mem::transmute_copy(&self.color_dx12.unwrap().1),
                std::ptr::null() as _,
                self.color_dx12.unwrap().0,
            )
        };
        unsafe {
            oidnSetFilterImage(
                self.handle,
                b"color\0" as *const _ as _,
                color_ptr as *mut _,
                OIDNFormat_OIDN_FORMAT_FLOAT3,
                self.img_dims.0 as _,
                self.img_dims.1 as _,
                0,
                pixelstride,
                bytes_per_row,
            );
        }

        if let Some(albedo) = self.albedo_dx12.as_ref() {
            let buffer = unsafe {
                oidnNewSharedBufferFromWin32Handle(
                    self.device.0,
                    OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                    std::mem::transmute_copy(&albedo.1),
                    std::ptr::null() as _,
                    albedo.0,
                )
            };
            unsafe {
                oidnSetFilterImage(
                    self.handle,
                    b"albedo\0" as *const _ as _,
                    buffer as *mut _,
                    OIDNFormat_OIDN_FORMAT_FLOAT3,
                    self.img_dims.0 as _,
                    self.img_dims.1 as _,
                    0,
                    pixelstride,
                    bytes_per_row,
                );
            }
        }

        if let Some(normal) = self.normal_dx12.as_ref() {
            let buffer = unsafe {
                oidnNewSharedBufferFromWin32Handle(
                    self.device.0,
                    OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                    std::mem::transmute_copy(&normal.1),
                    std::ptr::null() as _,
                    normal.0,
                )
            };
            unsafe {
                oidnSetFilterImage(
                    self.handle,
                    b"normal\0" as *const _ as _,
                    buffer as *mut _,
                    OIDNFormat_OIDN_FORMAT_FLOAT3,
                    self.img_dims.0 as _,
                    self.img_dims.1 as _,
                    0,
                    pixelstride,
                    bytes_per_row,
                );
            }
        }

        let output_buffer: *mut OIDNBufferImpl = unsafe {
            oidnNewSharedBufferFromWin32Handle(
                self.device.0,
                OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                std::mem::transmute(self.output_dx12.unwrap().1),
                std::ptr::null() as _,
                self.output_dx12.unwrap().0,
            )
        };
        unsafe {
            oidnSetFilterImage(
                self.handle,
                b"output\0" as *const _ as _,
                output_buffer as _,
                OIDNFormat_OIDN_FORMAT_FLOAT3,
                self.img_dims.0 as _,
                self.img_dims.1 as _,
                0,
                pixelstride,
                bytes_per_row,
            );
        }

        unsafe {
            oidnSetFilterBool(self.handle, b"hdr\0" as *const _ as _, self.hdr);
            oidnSetFilterFloat(
                self.handle,
                b"inputScale\0" as *const _ as _,
                self.input_scale,
            );
            oidnSetFilterBool(self.handle, b"srgb\0" as *const _ as _, self.srgb);
            oidnSetFilterInt(
                self.handle,
                b"quality\0" as *const _ as _,
                OIDNQuality_OIDN_QUALITY_BALANCED,
            );
            oidnSetFilterBool(self.handle, b"clean_aux\0" as *const _ as _, self.clean_aux);

            oidnCommitFilter(self.handle);
        }

        Ok(())
    }

    pub fn exec(&mut self) {
        unsafe {
            oidnExecuteFilter(self.handle);
        }
    }

    fn execute_filter(&self, color: Option<&[f32]>, output: &mut [f32]) -> Result<(), FilterError> {
        let pixelstride = size_of::<f32>() * 4;
        let bytes_per_row = 0;

        let color_ptr = unsafe {
            oidnNewSharedBufferFromWin32Handle(
                self.device.0,
                OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                std::mem::transmute_copy(&self.color_dx12.unwrap().1),
                std::ptr::null() as _,
                self.color_dx12.unwrap().0,
            )
        };

        unsafe {
            oidnSetFilterImage(
                self.handle,
                b"color\0" as *const _ as _,
                color_ptr as *mut _,
                OIDNFormat_OIDN_FORMAT_FLOAT3,
                self.img_dims.0 as _,
                self.img_dims.1 as _,
                0,
                pixelstride,
                bytes_per_row,
            );
        }

        let output_buffer: *mut OIDNBufferImpl = unsafe {
            oidnNewSharedBufferFromWin32Handle(
                self.device.0,
                OIDNExternalMemoryTypeFlag_OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                std::mem::transmute(self.output_dx12.unwrap().1),
                std::ptr::null() as _,
                self.output_dx12.unwrap().0,
            )
        };

        unsafe {
            oidnSetFilterImage(
                self.handle,
                b"output\0" as *const _ as _,
                output_buffer as _,
                OIDNFormat_OIDN_FORMAT_FLOAT3,
                self.img_dims.0 as _,
                self.img_dims.1 as _,
                0,
                pixelstride,
                bytes_per_row,
            );
        }

        unsafe {
            oidnSetFilterBool(self.handle, b"hdr\0" as *const _ as _, self.hdr);
            oidnSetFilterFloat(
                self.handle,
                b"inputScale\0" as *const _ as _,
                self.input_scale,
            );
            oidnSetFilterBool(self.handle, b"srgb\0" as *const _ as _, self.srgb);
            oidnSetFilterInt(
                self.handle,
                b"quality\0" as *const _ as _,
                OIDNQuality_OIDN_QUALITY_HIGH,
            );
            oidnSetFilterBool(self.handle, b"clean_aux\0" as *const _ as _, self.clean_aux);

            oidnCommitFilter(self.handle);
            //let buffer = vec![0u8; 1000];
            //let error = oidnGetDeviceError(self.device.0, &mut buffer.as_ptr() as *mut _ as *mut _);
            oidnExecuteFilter(self.handle);

            //println!("blah {:?}", error);
        }

        Ok(())
    }
}

impl<'a, 'b> Drop for RayTracing<'b> {
    fn drop(&mut self) {
        unsafe {
            oidnReleaseFilter(self.handle);
            oidnReleaseDevice(self.device.0);
        }
    }
}

unsafe impl<'a, 'b> Send for RayTracing<'b> {}
