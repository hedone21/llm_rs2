#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}
