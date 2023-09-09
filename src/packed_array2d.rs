use crate::{BlockType, PackedArray};

#[derive(Default, Clone, Hash, PartialEq, Eq)]
pub struct PackedArray2D {
    width: u8,
    inner: PackedArray,
}

impl PackedArray2D {
    pub fn from_slice(width: usize, height: usize, arr: &[u8]) -> Self {
        assert_eq!(width * height, arr.len());

        Self {
            width: width as _,
            inner: PackedArray::from(arr),
        }
    }

    pub fn width(&self) -> usize {
        self.width as _
    }

    pub fn height(&self) -> usize {
        self.inner.len() / self.width()
    }

    pub fn as_1d(&self) -> &PackedArray {
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
        y * self.width() + x
    }

    fn one_to_two(&self, i: usize) -> [usize; 2] {
        [i % self.width(), i / self.width()]
    }

    pub fn enumerate(&self) -> impl Iterator<Item = ([usize; 2], BlockType)> + '_ {
        self.inner
            .iter()
            .enumerate()
            .map(|(i, el)| (self.one_to_two(i), el))
    }

    pub fn get<I>(&self, [x, y]: [I; 2]) -> BlockType
    where
        I: num::cast::AsPrimitive<usize>,
    {
        let i = self.two_to_one([x.as_(), y.as_()]);
        self.inner.get(i)
    }

    pub fn set<I>(&mut self, [x, y]: [I; 2], v: BlockType)
    where
        I: num::cast::AsPrimitive<usize>,
    {
        let i = self.two_to_one([x.as_(), y.as_()]);
        self.inner.set(i, v)
    }
}

impl std::fmt::Debug for PackedArray2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedArray2D")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("inner", &self.inner)
            .finish()
    }
}
