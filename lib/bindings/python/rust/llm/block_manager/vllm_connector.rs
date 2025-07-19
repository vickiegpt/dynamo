use super::*;

use pyo3::wrap_pymodule;

use crate::llm::block_manager::BlockManager as PyBlockManager;

#[pymodule]
fn _vllm_connector_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvbmConnectorLeader>()?;
    Ok(())
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(_vllm_connector_integration))?;
    Ok(())
}

#[pyclass]
pub struct KvbmConnectorLeader {
    block_manager: PyBlockManager,
}

impl KvbmConnectorLeader {
    pub fn block_manager(&self) -> &VllmBlockManager {
        self.block_manager.get_block_manager()
    }
}

#[pymethods]
impl KvbmConnectorLeader {
    #[new]
    #[pyo3(signature = (block_manager))]
    pub fn new(block_manager: PyBlockManager) -> Self {
        Self {
            block_manager,
        }
    }

    
}