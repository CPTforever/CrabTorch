use std::{cell::RefCell, fmt::Display, ops::{Add, Div, Mul, Sub}, rc::Rc};

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
    fn get_data_index(&self, index: &[usize], tile: bool) -> Result<usize, TensorError> {
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
        let new_shape = self.shape[index.len()..].to_vec();
        let (new_size, _) = get_size_and_strides(&new_shape);
        Ok(Tensor {
            data: self.data.clone(),
            base_index: base_index,
            size: new_size,
            shape: new_shape,
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
            base_index: self.base_index,
            size: self.size,
            shape: new_shape.to_vec(),
            strides: strides
        })
    }

    pub fn flatten(&self) -> Result<Tensor<T>, TensorError> {
        self.reshape(&[self.size])
    }
}

impl<A> FromIterator<A> for Tensor<A>  {
    fn from_iter<T: IntoIterator<Item=A>>(iter: T) -> Self {
        let v: Vec<A> = iter.into_iter().collect();
        return Tensor {
            size: v.len(),
            shape: vec![v.len()],
            data: Rc::new(RefCell::new(v)),
            base_index: 0,
            strides: vec![1]
        };
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

    pub fn deep_clone(&self) -> Tensor<T> {
        let mut new_data = Vec::<T>::with_capacity(self.size);
        let new_shape =  self.shape.clone();
        for v in self.clone() {
            new_data.push(v);
        }
        let (size, strides) = get_size_and_strides(&new_shape);
        Tensor {
            data: Rc::new(RefCell::new(new_data)),
            base_index: 0,
            size: size,
            shape: new_shape,
            strides: strides
        }
    }
}

impl<T> Tensor<T>
where T: Clone + From<u32> + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>
{
    pub fn linspace(x0: T, xend: T, n: u32) -> Tensor<T> {
        let n_as_t = T::from(n);
        let dx = (xend - x0.clone()) / n_as_t;
        Self::from_iter((0..n).map(|x| x0.clone() + T::from(x) * dx.clone()))
    }
}

impl<T> Tensor<T> where Standard: Distribution<T> {
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

/* This is used internally to efficiently iterate through the
   data indices of a tensor in the correct order */
struct TensorIndexIterator<T> {
    pub tensor: Tensor<T>,
    index: Vec<usize>,
    data_index: usize,
    is_done: bool
}

impl<T> TensorIndexIterator<T> {
    fn new(tensor: Tensor<T>) -> TensorIndexIterator<T> {
        TensorIndexIterator {
            index: vec![0 as usize; tensor.rank()],
            data_index: tensor.base_index,
            tensor: tensor,
            is_done: false
        }
    }
}

impl<T> Iterator for TensorIndexIterator<T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }
        let result = self.data_index;
        for d in (0..self.tensor.rank()).rev() {
            self.index[d] += 1;
            self.data_index += self.tensor.strides[d];
            if self.index[d] >= self.tensor.shape[d] {
                self.index[d] = 0;
                self.data_index -= self.tensor.shape[d] * self.tensor.strides[d];
                if d == 0 {
                    self.is_done = true;
                }
            } else {
                break;
            }
        }
        Some(result)
    }
}

pub struct TensorIterator<T> {
    index_iterator: TensorIndexIterator<T>
}

impl<'a, T> TensorIterator<T> {
    pub fn new(tensor: Tensor<T>) -> TensorIterator<T> {
        TensorIterator {
            index_iterator: TensorIndexIterator::new(tensor)
        }
    }
}

impl<T: Clone> Iterator for TensorIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iterator.next() {
            Some(index) => {
                let data = self.index_iterator.tensor.data.borrow();
                Some(data[index].clone())
            },
            None => None
        }
    }
}

impl<T: Clone> IntoIterator for Tensor<T> {
    type Item = T;

    type IntoIter = TensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator::new(self)
    }
}

impl<T: Clone> IntoIterator for &Tensor<T> {
    type Item = T;

    type IntoIter = TensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator::new(self.clone())
    }
}