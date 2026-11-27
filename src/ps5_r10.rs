//! R10 research-only route tape. No inference policy or production promotion.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Route {
    pub layer: usize,
    pub pos: usize,
    pub tokens: Vec<u32>,
    pub top_k: usize,
    pub ids: Vec<i32>,
    pub weights: Vec<f32>,
}

pub(crate) fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x,y)| x.to_bits() == y.to_bits())
}

impl Route {
    pub fn validate(&self, layer: usize, pos: usize, tokens: &[u32], k: usize, experts: usize) -> Result<()> {
        ensure!(self.layer == layer && self.pos == pos && self.tokens == tokens && self.top_k == k, "R10 route tape coordinate mismatch");
        let n = tokens.len().checked_mul(k).ok_or_else(|| anyhow::anyhow!("R10 route size overflow"))?;
        ensure!(k > 0 && k <= experts && self.ids.len() == n && self.weights.len() == n, "R10 route tape dimensions");
        ensure!(self.ids.iter().all(|&e| e >= 0 && (e as usize) < experts) && self.weights.iter().all(|v| v.is_finite() && *v >= 0.0), "R10 invalid route values");
        ensure!(self.ids.chunks_exact(k).all(|row| row.iter().enumerate().all(|(i,e)| !row[..i].contains(e))), "R10 duplicate expert");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn route_validation() {
        let r = Route{layer:1,pos:32,tokens:vec![3,4],top_k:2,ids:vec![0,2,1,2],weights:vec![0.5,0.5,0.25,0.75]};
        assert!(r.validate(1,32,&[3,4],2,3).is_ok());
        assert!(r.validate(1,32,&[3,5],2,3).is_err());
        assert!(r.validate(1,0,&[3,4],2,3).is_err());
        for ids in [vec![0,0,1,2],vec![0,3,1,2],vec![0,-1,1,2],vec![0]] {
            assert!(Route{ids,..r.clone()}.validate(1,32,&[3,4],2,3).is_err());
        }
        assert!(Route{weights:vec![f32::NAN;4],..r.clone()}.validate(1,32,&[3,4],2,3).is_err());
        assert!(!same_bits(&[0.0],&[-0.0]));
        assert!(!same_bits(&[1.0],&[1.0,2.0]));
    }
}
