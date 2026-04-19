//! Boilerplate macro for forwarding `IMFAttributes_Impl` to a backing
//! `IMFAttributes` field named `attributes`.
//!
//! Both `VulvatarMediaSource` and `VulvatarMediaStream` need to expose
//! `IMFAttributes` as a first-class COM interface (Frame Server's FsProxy
//! QI's it during catalog matching), and both forward every call to a
//! backing store. The 28-method block is identical between them — keep
//! the boilerplate in one place so it can't drift.
//!
//! Usage: `forward_imfattributes!(VulvatarMediaSource_Impl);`

#[macro_export]
macro_rules! forward_imfattributes {
    ($impl_ty:ident) => {
        impl windows::Win32::Media::MediaFoundation::IMFAttributes_Impl for $impl_ty {
            fn GetItem(
                &self,
                guidkey: *const windows::core::GUID,
                pvalue: *mut windows::core::PROPVARIANT,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.GetItem(guidkey, Some(pvalue)) }
            }

            fn GetItemType(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<windows::Win32::Media::MediaFoundation::MF_ATTRIBUTE_TYPE>
            {
                unsafe { self.attributes.GetItemType(guidkey) }
            }

            fn CompareItem(
                &self,
                guidkey: *const windows::core::GUID,
                value: *const windows::core::PROPVARIANT,
            ) -> windows::core::Result<windows::Win32::Foundation::BOOL> {
                unsafe { self.attributes.CompareItem(guidkey, value) }
            }

            fn Compare(
                &self,
                ptheirs: Option<&windows::Win32::Media::MediaFoundation::IMFAttributes>,
                matchtype: windows::Win32::Media::MediaFoundation::MF_ATTRIBUTES_MATCH_TYPE,
            ) -> windows::core::Result<windows::Win32::Foundation::BOOL> {
                unsafe {
                    self.attributes
                        .Compare(ptheirs.unwrap_or(&self.attributes), matchtype)
                }
            }

            fn GetUINT32(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<u32> {
                unsafe { self.attributes.GetUINT32(guidkey) }
            }

            fn GetUINT64(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<u64> {
                unsafe { self.attributes.GetUINT64(guidkey) }
            }

            fn GetDouble(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<f64> {
                unsafe { self.attributes.GetDouble(guidkey) }
            }

            fn GetGUID(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<windows::core::GUID> {
                unsafe { self.attributes.GetGUID(guidkey) }
            }

            fn GetStringLength(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<u32> {
                unsafe { self.attributes.GetStringLength(guidkey) }
            }

            fn GetString(
                &self,
                guidkey: *const windows::core::GUID,
                pwszvalue: windows::core::PWSTR,
                cchbufsize: u32,
                pcchlength: *mut u32,
            ) -> windows::core::Result<()> {
                unsafe {
                    let slice = core::slice::from_raw_parts_mut(pwszvalue.0, cchbufsize as usize);
                    self.attributes.GetString(guidkey, slice, Some(pcchlength))
                }
            }

            fn GetAllocatedString(
                &self,
                guidkey: *const windows::core::GUID,
                ppwszvalue: *mut windows::core::PWSTR,
                pcchlength: *mut u32,
            ) -> windows::core::Result<()> {
                unsafe {
                    self.attributes
                        .GetAllocatedString(guidkey, ppwszvalue, pcchlength)
                }
            }

            fn GetBlobSize(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<u32> {
                unsafe { self.attributes.GetBlobSize(guidkey) }
            }

            fn GetBlob(
                &self,
                guidkey: *const windows::core::GUID,
                pbuf: *mut u8,
                cbbufsize: u32,
                pcbblobsize: *mut u32,
            ) -> windows::core::Result<()> {
                unsafe {
                    let slice = core::slice::from_raw_parts_mut(pbuf, cbbufsize as usize);
                    self.attributes.GetBlob(guidkey, slice, Some(pcbblobsize))
                }
            }

            fn GetAllocatedBlob(
                &self,
                guidkey: *const windows::core::GUID,
                ppbuf: *mut *mut u8,
                pcbsize: *mut u32,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.GetAllocatedBlob(guidkey, ppbuf, pcbsize) }
            }

            fn GetUnknown(
                &self,
                guidkey: *const windows::core::GUID,
                riid: *const windows::core::GUID,
                ppv: *mut *mut core::ffi::c_void,
            ) -> windows::core::Result<()> {
                unsafe {
                    let vt = windows::Win32::Media::MediaFoundation::IMFAttributes::vtable(
                        &self.attributes,
                    );
                    (vt.GetUnknown)(
                        windows::Win32::Media::MediaFoundation::IMFAttributes::as_raw(
                            &self.attributes,
                        ),
                        guidkey,
                        riid,
                        ppv,
                    )
                    .ok()
                }
            }

            fn SetItem(
                &self,
                guidkey: *const windows::core::GUID,
                value: *const windows::core::PROPVARIANT,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetItem(guidkey, value) }
            }

            fn DeleteItem(
                &self,
                guidkey: *const windows::core::GUID,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.DeleteItem(guidkey) }
            }

            fn DeleteAllItems(&self) -> windows::core::Result<()> {
                unsafe { self.attributes.DeleteAllItems() }
            }

            fn SetUINT32(
                &self,
                guidkey: *const windows::core::GUID,
                unvalue: u32,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetUINT32(guidkey, unvalue) }
            }

            fn SetUINT64(
                &self,
                guidkey: *const windows::core::GUID,
                unvalue: u64,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetUINT64(guidkey, unvalue) }
            }

            fn SetDouble(
                &self,
                guidkey: *const windows::core::GUID,
                fvalue: f64,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetDouble(guidkey, fvalue) }
            }

            fn SetGUID(
                &self,
                guidkey: *const windows::core::GUID,
                guidvalue: *const windows::core::GUID,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetGUID(guidkey, guidvalue) }
            }

            fn SetString(
                &self,
                guidkey: *const windows::core::GUID,
                wszvalue: &windows::core::PCWSTR,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetString(guidkey, *wszvalue) }
            }

            fn SetBlob(
                &self,
                guidkey: *const windows::core::GUID,
                pbuf: *const u8,
                cbbufsize: u32,
            ) -> windows::core::Result<()> {
                unsafe {
                    let slice = core::slice::from_raw_parts(pbuf, cbbufsize as usize);
                    self.attributes.SetBlob(guidkey, slice)
                }
            }

            fn SetUnknown(
                &self,
                guidkey: *const windows::core::GUID,
                punknown: Option<&windows::core::IUnknown>,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.SetUnknown(guidkey, punknown) }
            }

            fn LockStore(&self) -> windows::core::Result<()> {
                unsafe { self.attributes.LockStore() }
            }

            fn UnlockStore(&self) -> windows::core::Result<()> {
                unsafe { self.attributes.UnlockStore() }
            }

            fn GetCount(&self) -> windows::core::Result<u32> {
                unsafe { self.attributes.GetCount() }
            }

            fn GetItemByIndex(
                &self,
                unindex: u32,
                pguidkey: *mut windows::core::GUID,
                pvalue: *mut windows::core::PROPVARIANT,
            ) -> windows::core::Result<()> {
                unsafe {
                    self.attributes
                        .GetItemByIndex(unindex, pguidkey, Some(pvalue))
                }
            }

            fn CopyAllItems(
                &self,
                pdest: Option<&windows::Win32::Media::MediaFoundation::IMFAttributes>,
            ) -> windows::core::Result<()> {
                unsafe { self.attributes.CopyAllItems(pdest) }
            }
        }
    };
}
