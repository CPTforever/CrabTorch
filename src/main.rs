mod tensor;
mod error;

use error::TensorError;
use tensor::Tensor;

fn main() -> Result<(), TensorError> {
    let t = Tensor::from_array([1, 2, 3, 4]).reshape([2, 2])?;
    println!("{}", t);
    println!("{}", t.get(&[1, 1])?);
    println!("{}", Tensor::scalar(1));
    Ok(())
}
