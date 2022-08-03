#[derive(Default, Clone, Hash, PartialEq, Eq)]
pub struct Array2D<T> {
    width: usize,
    inner: Vec<T>,
}

impl<T: Default> Array2D<T> {
    pub fn empty(width: usize, height: usize) -> Self {
        Self {
            width,
            inner: (0..width * height).map(|_| Default::default()).collect(),
        }
    }
}

impl<T: Clone> Array2D<T> {
    pub fn with_element(width: usize, height: usize, element: T) -> Self {
        Self {
            width,
            inner: vec![element; width * height],
        }
    }
}

impl<T> Array2D<T> {
    pub fn from_vec(width: usize, height: usize, arr: Vec<T>) -> Self {
        assert_eq!(width * height, arr.len());

        Self { width, inner: arr }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.inner.len() / self.width()
    }

    pub fn as_1d(&self) -> &[T] {
        &self.inner
    }

    pub fn swap<I>(&mut self, a: [I; 2], b: [I; 2])
    where
        I: num::cast::AsPrimitive<usize>,
    {
        let a = self.two_to_one([a[0].as_(), a[1].as_()]);
        let b = self.two_to_one([b[0].as_(), b[1].as_()]);
        self.inner.swap(a, b)
    }

    fn two_to_one(&self, [x, y]: [usize; 2]) -> usize {
        y * self.width + x
    }

    fn one_to_two(&self, i: usize) -> [usize; 2] {
        [i % self.width(), i / self.width()]
    }

    pub fn enumerate(&self) -> impl Iterator<Item = ([usize; 2], &T)> {
        self.inner
            .iter()
            .enumerate()
            .map(|(i, el)| (self.one_to_two(i), el))
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Array2D<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array2D")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("inner", &self.inner)
            .finish()
    }
}

impl<T, I> std::ops::Index<[I; 2]> for Array2D<T>
where
    I: num::cast::AsPrimitive<usize>,
{
    type Output = T;
    fn index(&self, [x, y]: [I; 2]) -> &Self::Output {
        let i = self.two_to_one([x.as_(), y.as_()]);
        &self.inner[i]
    }
}

impl<T, I> std::ops::IndexMut<[I; 2]> for Array2D<T>
where
    I: num::cast::AsPrimitive<usize>,
{
    fn index_mut(&mut self, [x, y]: [I; 2]) -> &mut Self::Output {
        let i = self.two_to_one([x.as_(), y.as_()]);
        &mut self.inner[i]
    }
}
