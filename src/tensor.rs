use std::{cell::RefCell, fmt::Display, rc::Rc};

use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::error::TensorError;

#[derive(Clone)]
pub struct Tensor<T> {
    data: Rc<RefCell<Vec<T>>>,
    shape: Vec<usize>,
    strides: Vec<usize>
}

fn get_size_and_strides(shape: &[usize]) -> (usize, Vec<usize>) {
    let mut strides = Vec::<usize>::with_capacity(shape.len());
    let mut size: usize = 1;
    for d in shape {
        strides.push(size);
        size *= d;
    }
    return (size, strides);
}

impl<T> Tensor<T> {
    pub fn reshape<S: Into<Vec<usize>>>(&self, new_shape: S) -> Result<Tensor<T>, TensorError> {
        let _shape: Vec<usize> = new_shape.into();
        let (size, strides) = get_size_and_strides(_shape.as_slice());
        let (current_size, _) = get_size_and_strides(self.shape.as_slice());
        if size != current_size {
            return Err(TensorError::new("new shape cannot be of a different size"))
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: _shape,
            strides: strides
        })
    }
}

impl<T: Clone> Tensor<T> {
    pub fn from_shape<S: Into<Vec<usize>>>(value: T, shape: S) -> Tensor<T> {
        let _shape: Vec<usize> = shape.into();
        let (size, strides) = get_size_and_strides(_shape.as_slice());
        Tensor {
            data: Rc::new(RefCell::new(vec![value; size])),
            strides: strides,
            shape: _shape
        }
    }

    pub fn from_array<A: Into<Vec<T>>>(arr: A) -> Tensor<T> {
        let _arr = arr.into();
        Tensor {
            strides: vec![1],
            shape: vec![_arr.len()],
            data: Rc::new(RefCell::new(_arr))
        }
    }
}

impl<T> Tensor<T> 
    where
    Standard: Distribution<T>,{
    
    pub fn rand<S: Into<Vec<usize>>>(shape: S) -> Tensor<T> {
        let _shape: Vec<usize> = shape.into();
        let (size, strides) = get_size_and_strides(_shape.as_slice());
        let mut rng = rand::thread_rng();

        Tensor {
            data: Rc::new(RefCell::new((0..size).map(|_| rng.gen()).collect())),
            strides: strides,
            shape: _shape
        }
    } 
}

impl<T: Display> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.borrow();
        let mut depth = 0;
        for i in 0..data.len() {
            let mut open = if i == 0 { 1 } else { 0 };
            let mut close = if i + 1 == data.len() { 1 } else { 0 };
            for s in &self.strides[1..] {
                if i % s == 0 {
                    open += 1;
                }
            }
            for s in &self.strides[1..] {
                if (i + 1) % s == 0 {
                    close += 1;
                }
            }
            for _ in 0..open {
                write!(f, "[")?;
            }
            write!(f, "{}", data[i])?;
            if close == 0 {
                write!(f, ", ")?;
            }
            for _ in 0..close {
                write!(f, "]")?;
            }
            depth += open - close;
            if close > 0 && depth > 0 {
                write!(f, "\n")?;
                for _ in 0..depth {
                    write!(f, " ")?;
                }
            }
        }
        Ok(())
    }
}