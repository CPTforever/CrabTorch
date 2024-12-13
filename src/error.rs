use std::fmt::{Debug, Display};

// Super basic error struct, can make better later
pub struct TensorError {
    pub message: String
}

impl TensorError {
    pub fn new<M: Into<String>>(message: M) -> TensorError {
        TensorError {
            message: message.into()
        }
    }
}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        Ok(())
    }
}

impl Debug for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        Ok(())
    }
}