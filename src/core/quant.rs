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
        for i in 0..QK8_0 {
            out[i] = self.qs[i] as f32 * d;
        }
    }
}
