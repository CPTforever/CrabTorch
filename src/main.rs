mod tensor;
mod error;

use error::TensorError;
use tensor::Tensor;

fn main() -> Result<(), TensorError> {
    let t = Tensor::<f32>::rand(&[10, 10]).reshape(&[25, 4])?;
    println!("{}", t);
    let mut i = 0;
    for n in t {
        println!("{}: {}", i, n);
        i += 1;
    }
    Ok(())
}
