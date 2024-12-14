use std::{cell::RefCell, fmt::Display, rc::Rc};

use num::CheckedAdd;
use num_traits::AsPrimitive;
use rand::{distributions::Standard, prelude::Distribution, Rng};

use crate::error::TensorError;

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Rc<RefCell<Vec<T>>>,
    base_index: usize,
    size: usize,
    shape: Vec<usize>,
    strides: Vec<usize>
}

fn get_size_and_strides(shape: &[usize]) -> (usize, Vec<usize>) {
    let mut strides = vec![0 as usize; shape.len()];
    let mut size: usize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = size;
        size *= shape[i];
    }
    return (size, strides);
}

impl<T> Tensor<T> {
    pub fn get_data_index(&self, index: &[usize], tile: bool) -> Result<usize, TensorError> {
        if index.len() > self.shape.len() {
            return Err(TensorError::new("index has too many dimensions"));
        }
        let mut data_index: usize = self.base_index;
        for i in 0..index.len() {
            let mut next_index = index[i];
            if tile {
                next_index %= self.shape[i];
            } else if next_index >= self.shape[i] {
                return Err(TensorError::new(format!("index {} is out of range for dimension {}", next_index, i)));
            }
            data_index += next_index * self.strides[i];
        }
        Ok(data_index)
    }

    pub fn get(&self, index: &[usize]) -> Result<Tensor<T>, TensorError> {
        let base_index = self.get_data_index(index, false)?;
        Ok(Tensor {
            data: self.data.clone(),
            base_index: base_index,
            size: self.strides[index.len() - 1],
            shape: self.shape[index.len()..].to_vec(),
            strides: self.strides[index.len()..].to_vec()
        })
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor<T>, TensorError> {
        let (size, strides) = get_size_and_strides(new_shape);
        if size != self.size {
            return Err(TensorError::new("new shape cannot be of a different size"))
        }
        Ok(Tensor {
            data: self.data.clone(),
            base_index: 0,
            size: self.size,
            shape: new_shape.to_vec(),
            strides: strides
        })
    }
}

impl<T: Clone> Tensor<T> {
    pub fn from_shape(value: T, shape: &[usize]) -> Tensor<T> {
        let (size, strides) = get_size_and_strides(shape);
        Tensor {
            data: Rc::new(RefCell::new(vec![value; size])),
            base_index: 0,
            size: size,
            strides: strides,
            shape: shape.to_vec()
        }
    }

    pub fn from_array(arr: &[T]) -> Tensor<T> {
        Tensor {
            base_index: 0,
            strides: vec![1],
            shape: vec![arr.len()],
            size: arr.len(),
            data: Rc::new(RefCell::new(arr.to_vec()))
        }
    }

    pub fn scalar(value: T) -> Tensor<T> {
        Self::from_shape(value, &[])
    }
}

// This is like gay people in the military, don't ask don't tell
impl<T> Tensor<T> where 
    T: num_traits::cast::AsPrimitive<T>,
    isize: num_traits::cast::AsPrimitive<T> {
    pub fn step_range(range: num::iter::RangeStep<isize>) -> Tensor<T> {
        let voo: Vec<T> = range.map(|x| isize::as_(x)).collect();
        let _shape = vec![voo.len()];
        let (size, strides) = get_size_and_strides(&_shape);

        let foo = Rc::new(RefCell::new(voo));
        Tensor {
            data: foo,
            strides: strides,
            shape: _shape,
            base_index: 0,
            size,
        }
    }

    pub fn arange(range: std::ops::Range<isize>) -> Tensor<T> {
        let voo: Vec<T> = range.map(|x| isize::as_(x)).collect();
        let _shape = vec![voo.len()];
        let (size, strides) = get_size_and_strides(&_shape);

        let foo = Rc::new(RefCell::new(voo));
        Tensor {
            data: foo,
            strides: strides,
            shape: _shape,
            base_index: 0,
            size,
        }
    }
}

// Be careful with this, I stole it.
impl<T: num_traits::Float + From<u32>> Tensor<T> {
    pub fn linspace(x0: T, xend: T, n: u32) -> Tensor<T> {
        let to_float = |i: u32| i.try_into().unwrap_or_else(|_| panic!());
        let dx = (xend - x0) / to_float(n - 1);
        let vec: Vec<T> = (0..n).map(|i| x0 + to_float(i) * dx).collect();

        let _shape = vec![vec.len()];
        let (size, strides) = get_size_and_strides(&_shape);
        let foo = Rc::new(RefCell::new(vec));
        Tensor {
            data: foo,
            strides: strides,
            shape: _shape,
            base_index: 0,
            size,
        }
    }
}

impl<T> Tensor<T> 
    where
    Standard: Distribution<T>,{
    
    pub fn rand(shape: &[usize]) -> Tensor<T> {
        let (size, strides) = get_size_and_strides(shape);
        let mut rng = rand::thread_rng();
        Tensor {
            data: Rc::new(RefCell::new((0..size).map(|_| rng.gen()).collect())),
            base_index: 0,
            size: size,
            strides: strides,
            shape: shape.to_vec()
        }
    } 
}

impl<T: Display> Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.borrow();

        if self.is_scalar() {
            write!(f, "{}", data[self.base_index])?;
            return Ok(());
        }

        let mut depth = 0;
        for i in 0..self.size {
            let mut open = if i == 0 { 1 } else { 0 };
            let mut close = if i + 1 == self.size { 1 } else { 0 };
            for s in &self.strides[..self.strides.len() - 1] {
                if i % s == 0 {
                    open += 1;
                }
            }
            for s in &self.strides[..self.strides.len() - 1] {
                if (i + 1) % s == 0 {
                    close += 1;
                }
            }
            for _ in 0..open {
                write!(f, "[")?;
            }
            write!(f, "{}", data[self.base_index + i])?;
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