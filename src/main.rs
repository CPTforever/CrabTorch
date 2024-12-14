mod tensor;
mod error;

use error::TensorError;
use tensor::Tensor;

fn main() -> Result<(), TensorError> {
    let t = Tensor::from_array(&[1, 2, 3, 4]).reshape(&[2, 2])?;
    println!("{}", t);
    println!("{}", t.get(&[1, 1])?);
    println!("{}", Tensor::scalar(1));
    let zeros = Tensor::from_shape(0f32, &[2, 2, 2]);
    let array = Tensor::from_array(&[1, 2, 3, 4, 5, 6]);
    println!("{}", zeros);
    println!("{}", array);
    println!();

    let reshaped = array.reshape(&[2, 3])?;
    let verical = array.reshape(&[1, 6])?;
    println!("{}", array);
    println!("{}", reshaped);
    println!("{}", verical);

    let zeros = Tensor::<f32>::rand(&[8, 8, 8]);
    println!("zeros: {}", zeros);
    Ok(())
}
