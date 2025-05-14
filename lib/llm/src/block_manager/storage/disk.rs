use super::*;

use std::fs::File;

use memmap2::{MmapMut, MmapOptions};

#[derive(Debug)]
pub struct DiskStorage {
    _file: File,
    size: usize,
    mmap_file: MmapMut,
    handles: RegistrationHandles,
}

impl Local for DiskStorage {}
impl SystemAccessible for DiskStorage {}

impl DiskStorage {
    pub fn new(size: usize) -> Result<Self, StorageError> {
        let file = tempfile::tempfile().map_err(|_| {
            StorageError::AllocationFailed("Unable to create tempfile.".to_string())
        })?;

        file.set_len(size as u64)
            .map_err(|_| StorageError::AllocationFailed("Unable to set file size.".to_string()))?;

        let mmap_file;

        unsafe {
            mmap_file = MmapOptions::new()
                .len(size)
                .huge(Some(21)) // Use 2MB pages for better perf.
                .populate()
                .map_mut(&file)
                .map_err(|_| StorageError::AllocationFailed("Unable to mmap file.".to_string()))?;
        };

        Ok(Self {
            _file: file,
            size,
            mmap_file,
            handles: RegistrationHandles::new(),
        })
    }
}

impl Storage for DiskStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::Disk
    }

    fn addr(&self) -> u64 {
        self.mmap_file.as_ptr() as u64
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.mmap_file.as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap_file.as_mut_ptr()
    }
}

impl RegisterableStorage for DiskStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

pub struct DiskAllocator;

impl StorageAllocator<DiskStorage> for DiskAllocator {
    fn allocate(&self, size: usize) -> Result<DiskStorage, StorageError> {
        DiskStorage::new(size)
    }
}
