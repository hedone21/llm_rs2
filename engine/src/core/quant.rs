use half::f16;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; QK4_0 / 2],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub d: f16,
    pub m: f16,
    pub qs: [u8; QK4_1 / 2],
}

// Ensure sizes are correct
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

// Helper for dequantization
impl BlockQ4_0 {
    pub fn dequantize(&self, out: &mut [f32; QK4_0]) {
        let d = self.d.to_f32();
        for i in 0..(QK4_0 / 2) {
            let b = self.qs[i];
            let v0 = (b & 0x0F) as i8 - 8;
            let v1 = (b >> 4) as i8 - 8;

            out[i] = v0 as f32 * d;
            out[i + QK4_0 / 2] = v1 as f32 * d;
        }
    }
}

impl BlockQ4_1 {
    pub fn dequantize(&self, out: &mut [f32; QK4_1]) {
        let d = self.d.to_f32();
        let m = self.m.to_f32();
        for i in 0..(QK4_1 / 2) {
            let b = self.qs[i];
            let v0 = (b & 0x0F) as f32;
            let v1 = (b >> 4) as f32;

            out[i] = v0 * d + m;
            out[i + QK4_1 / 2] = v1 * d + m;
        }
    }
}

pub const QK8_0: usize = 32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub d: f16,
    pub qs: [i8; QK8_0],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

impl BlockQ8_0 {
    pub fn dequantize(&self, out: &mut [f32; QK8_0]) {
        let d = self.d.to_f32();
        for (i, o) in out.iter_mut().enumerate().take(QK8_0) {
            *o = self.qs[i] as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_q4_0_dequantize() {
        let mut out = [0.0; QK4_0];

        // Scale = 2.0 (f16 corresponding to 2.0 is roughly 0x4000)
        let mut block = BlockQ4_0 {
            d: f16::from_f32(2.0),
            qs: [0; QK4_0 / 2],
        };

        // Let's set some specific nibbles
        // b = 0x1A (0b0001_1010)
        // v0 = 0x0A (10) -> 10 - 8 = 2
        // v1 = 0x01 (1) -> 1 - 8 = -7
        block.qs[0] = 0x1A;

        // Max nibble: 0xFF
        // v0 = 15 - 8 = 7
        // v1 = 15 - 8 = 7
        block.qs[1] = 0xFF;

        // Min nibble: 0x00
        // v0 = 0 - 8 = -8
        // v1 = 0 - 8 = -8
        block.qs[2] = 0x00;

        block.dequantize(&mut out);

        // Verification for qs[0] -> block 0 and 16
        assert_eq!(out[0], 2.0 * 2.0); // 4.0
        assert_eq!(out[16], -7.0 * 2.0); // -14.0

        // Verification for qs[1] -> block 1 and 17
        assert_eq!(out[1], 7.0 * 2.0); // 14.0
        assert_eq!(out[17], 7.0 * 2.0); // 14.0

        // Verification for qs[2] -> block 2 and 18
        assert_eq!(out[2], -8.0 * 2.0); // -16.0
        assert_eq!(out[18], -8.0 * 2.0); // -16.0

        // Unset ones should be (0 - 8) * 2.0 = -16.0
        assert_eq!(out[3], -16.0);
    }

    #[test]
    fn test_block_q4_0_zero_scale() {
        let mut out = [0.0; QK4_0];
        let block = BlockQ4_0 {
            d: f16::from_f32(0.0),
            qs: [0x55; QK4_0 / 2], // Some non-zero pattern
        };
        block.dequantize(&mut out);
        for val in out {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_block_q4_1_dequantize() {
        let mut out = [0.0; QK4_1];

        // Scale = 2.0, Min = -5.0
        let mut block = BlockQ4_1 {
            d: f16::from_f32(2.0),
            m: f16::from_f32(-5.0),
            qs: [0; QK4_1 / 2],
        };

        // b = 0x1A
        // v0 = 0x0A (10)
        // v1 = 0x01 (1)
        block.qs[0] = 0x1A;

        block.dequantize(&mut out);

        // out = v * d + m
        // out[0] = 10 * 2.0 - 5.0 = 15.0
        // out[16] = 1 * 2.0 - 5.0 = -3.0
        assert_eq!(out[0], 15.0);
        assert_eq!(out[16], -3.0);
    }

    #[test]
    fn test_block_q8_0_dequantize() {
        let mut out = [0.0; QK8_0];

        let mut block = BlockQ8_0 {
            d: f16::from_f32(0.5),
            qs: [0; QK8_0],
        };

        block.qs[0] = 10;
        block.qs[1] = -5;
        block.qs[31] = 127; // max i8

        block.dequantize(&mut out);

        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], -2.5);
        assert_eq!(out[31], 63.5);
    }

    #[test]
    fn test_struct_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn test_block_q4_1_zero_scale() {
        let mut out = [0.0; QK4_1];
        let block = BlockQ4_1 {
            d: f16::from_f32(0.0),
            m: f16::from_f32(3.0),
            qs: [0xAB; QK4_1 / 2], // Non-zero nibbles
        };
        block.dequantize(&mut out);
        // With d=0, all values should equal m (=3.0): v * 0.0 + 3.0 = 3.0
        for val in out {
            assert_eq!(
                val, 3.0,
                "Q4_1 zero scale should produce m for all elements"
            );
        }
    }
}
