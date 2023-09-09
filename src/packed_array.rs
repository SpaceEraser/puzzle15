pub type BlockType = u8;
type DoubleBlockType = u16;
const BS: usize = std::mem::size_of::<BlockType>() * 8;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct PackedArray {
    element_size: u8,
    len: u8,
    blocks: Box<[BlockType]>,
}

impl PackedArray {
    pub fn from_slice(element_size: u8, vec: &[BlockType]) -> Self {
        debug_assert!(element_size <= BS as u8);

        let num_chunks = (vec.len() * element_size as usize) as f64 / BS as f64;
        let mut this = Self {
            element_size,
            len: vec.len() as _,
            blocks: vec![0; num_chunks.ceil() as _].into_boxed_slice(),
        };
        for (i, &v) in vec.iter().enumerate() {
            this.set(i, v);
        }
        this
    }

    pub fn set(&mut self, i: usize, v: BlockType) {
        if cfg!(debug_assertions) && i >= self.len() {
            panic!(
                "index out of bounds: the len is {} but the index is {i}",
                self.len()
            );
        }

        let es = self.element_size as usize;
        let block_i = (i * es) / BS;
        let j = (i * es) % BS;

        let double_block = set_bits_in_range(self.get_double_block(block_i), v as _, j..j + es);
        self.blocks[block_i] = double_block as BlockType;

        if block_i + 1 < self.blocks.len() {
            self.blocks[block_i + 1] = (double_block >> BS) as BlockType;
        }
    }

    pub fn get(&self, i: usize) -> BlockType {
        if cfg!(debug_assertions) && i >= self.len() {
            panic!(
                "index out of bounds: the len is {} but the index is {i}",
                self.len()
            );
        }

        let es = self.element_size as usize;
        let block_i = (i * es) / BS;
        let j = (i * es) % BS;

        // dbg!(es, block_i, j, BS);
        // println!("double_block = 0b{:b}", self.get_double_block(block_i));

        get_bits_in_range(self.get_double_block(block_i), j..j + es) as BlockType
    }

    fn get_double_block(&self, i: usize) -> DoubleBlockType {
        self.blocks[i] as DoubleBlockType
            | ((self.blocks.get(i + 1).copied().unwrap_or(0) as DoubleBlockType) << BS)
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        let tmp = self.get(a);
        self.set(a, self.get(b));
        self.set(b, tmp);
    }

    pub fn len(&self) -> usize {
        self.len as _
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = BlockType> + '_ {
        (0..self.len).map(|i| self.get(i as _))
    }

    pub fn to_vec(&self) -> Vec<BlockType> {
        self.iter().collect()
    }
}

impl Default for PackedArray {
    fn default() -> Self {
        Self::from_slice(1, &[])
    }
}

impl From<&[BlockType]> for PackedArray {
    fn from(vec: &[BlockType]) -> Self {
        let leading_zeros = vec
            .iter()
            .map(|v| v.leading_zeros())
            .min()
            .unwrap_or(BS as _);
        Self::from_slice(BS as u8 - leading_zeros as u8, vec)
    }
}

#[inline]
fn set_bits_in_range<T>(block: T, v: T, r: std::ops::Range<usize>) -> T
where
    T: num::PrimInt,
{
    (block & !ones_in_range::<T>(r.clone()))
        | ((v & ones_in_range::<T>(0..r.end - r.start)) << r.start)
}

#[inline]
fn get_bits_in_range<T>(block: T, r: std::ops::Range<usize>) -> T
where
    T: num::PrimInt,
{
    (block & ones_in_range(r.clone())) >> r.start
}

#[inline]
#[allow(clippy::just_underscores_and_digits)]
fn ones_in_range<T>(r: std::ops::Range<usize>) -> T
where
    T: num::PrimInt,
{
    let _1 = num::one::<T>();
    ((_1 << (r.end - r.start)) - _1) << r.start
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn packed_array_from() {
        let p = PackedArray::from(&[0b001 as BlockType, 0b100] as &[BlockType]);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), 0b001);
        assert_eq!(p.get(1), 0b100);
    }

    #[test]
    fn packed_array_from_slice() {
        let p = PackedArray::from_slice(3, &[0b001, 0b100]);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get(0), 0b001);
        assert_eq!(p.get(1), 0b100);
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn packed_array_get() {
        let p = PackedArray {
            element_size: 6,
            len: 3,
            blocks: vec![0b00_000010, 0b1111_0000, 0b11].into_boxed_slice(),
        };
        assert_eq!(p.get(0), 0b000010);
        assert_eq!(p.get(1), 0b000000);
        assert_eq!(p.get(2), 0b111111);
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn packed_array_set() {
        let mut p = PackedArray {
            element_size: 6,
            len: 2,
            blocks: vec![0, 0].into_boxed_slice(),
        };
        p.set(1, 0b101010);
        assert_eq!(&*p.blocks, &[0b10_000000, 0b1010]);
        p.set(0, 0b000010);
        assert_eq!(&*p.blocks, &[0b10_000010, 0b1010]);
    }

    #[test]
    fn packed_array_1() {
        let p = PackedArray {
            element_size: 1,
            len: 4,
            blocks: vec![0b1101].into_boxed_slice(),
        };
        assert_eq!(p.get(0), 0b1);
        assert_eq!(p.get(1), 0b0);
        assert_eq!(p.get(2), 0b1);
        assert_eq!(p.get(3), 0b1);
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn packed_array_swap() {
        let mut p = PackedArray {
            element_size: 6,
            len: 2,
            blocks: vec![0b10_000000, 0b1010].into_boxed_slice(),
        };
        p.swap(0, 1);
        assert_eq!(&*p.blocks, &[0b00_101010, 0b0000]);
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn packed_array_len() {
        let p = PackedArray {
            element_size: 6,
            len: 2,
            blocks: vec![0b10_000000, 0b1010].into_boxed_slice(),
        };
        assert_eq!(p.len(), 2);
        let p = PackedArray {
            element_size: 1,
            len: 2,
            blocks: vec![0b10].into_boxed_slice(),
        };
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_ones_in_range() {
        {
            let actual = ones_in_range::<u32>(0..0);
            let expected = 0b0;
            assert_eq!(
                actual, expected,
                "expected 0b{expected:b}, got 0b{actual:b}"
            );
        }
        {
            let actual = ones_in_range::<u32>(1..1);
            let expected = 0b0;
            assert_eq!(
                actual, expected,
                "expected 0b{expected:b}, got 0b{actual:b}"
            );
        }
        {
            let actual = ones_in_range::<u32>(0..1);
            let expected = 0b01;
            assert_eq!(
                actual, expected,
                "expected 0b{expected:b}, got 0b{actual:b}"
            );
        }
        {
            let actual = ones_in_range::<u32>(1..3);
            let expected = 0b0110;
            assert_eq!(
                actual, expected,
                "expected 0b{expected:b}, got 0b{actual:b}"
            );
        }
        {
            let actual = ones_in_range::<u32>(16..32);
            let expected = 0b1111_1111_1111_1111_0000_0000_0000_0000;
            assert_eq!(
                actual, expected,
                "expected 0b{expected:b}, got 0b{actual:b}"
            );
        }
    }
}
