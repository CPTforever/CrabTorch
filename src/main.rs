mod tensor;
mod error;

use error::TensorError;
use tensor::Tensor;

fn main() -> Result<(), TensorError> {
    let t = Tensor::from_iter(0..100).reshape(&[50, 2])?; 

    let t2 = t.deep_clone();
    println!("{}", t);
    println!("{}", t2);
    Ok(())
}
