#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::time::Instant;

// use ark_bls12_381::{Bls12_381, FrParameters};
// use ark_ec::PairingEngine;
// use ark_bls12_381::{Bls12_381, FrParameters};
use ark_bls12_381::{Fr, FrConfig};

use ark_std::rc::Rc;

use ark_ff::{Field, Fp, MontBackend};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, Polynomial, SparseMultilinearExtension};
use ark_sumcheck::data_structures::{DenseOrSparseMultilinearExtension, ListOfProductsOfPolynomials};
use ark_sumcheck::gkr_round_sumcheck::data_structures::GKRProof;
use ark_sumcheck::gkr_round_sumcheck::GKRRoundSumcheck;
use ark_sumcheck::matmul::{ProverMatMul, VerifierMatMul};
use ark_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_sumcheck::{log2i, rbin, utils::matrix_to_mle};
use ark_sumcheck::rng::{Blake2b512Rng, FeedableRNG};
use ark_sumcheck::conv::VerifierConv2D;
use ark_sumcheck::ipformlsumcheck::prover::ProverState;
use ark_std::UniformRand;
use ark_sumcheck::mlsumcheck::MLSC;

use csv;
use csv::StringRecord;

fn read_matrix(filename: &str) -> std::result::Result<Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>>, Box<dyn std::error::Error>>{

    let mut reader = csv::ReaderBuilder::new()
    .has_headers(false)
    .from_path(filename)?;
    // let mut reader = csv::Reader::from_path(filename)?;
    let n_r = reader.records().count();

    let mut reader2 = csv::ReaderBuilder::new()
    .has_headers(false)
    .from_path(filename)?;

    let start = reader2.position().clone();

    let mut record = StringRecord::new();

    if reader2.read_record(&mut record)? {
        let n_c = record.len();
        
        // println!("Rows:{}", n_r);
        // println!("Cols:{}", n_c);

        let mut mat = vec![vec![Fr::from(0); n_c]; n_r];
        let mut cr = 0;

        reader2.seek(start)?;
        for result in reader2.records() {
            let record = result?;
            for i in 0..record.len() {
                mat[cr][i] = Fr::from((record.get(i).unwrap()).parse::<i128>().unwrap());
            }
            cr = cr + 1;
        }
        // println!("Check:{}", cr);
        return Ok(mat);
    } else {
        return Err(From::from("expected at least one record but got none"))
    }

}

pub fn main()
{
    std::env::set_var("RUST_BACKTRACE", "1");

    let cwd = std::env::current_dir().unwrap().into_os_string().into_string().unwrap();

    // ===================   ADDITION OF SQUARED   =====================================


    let sub_mat1: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/sq_mat1.csv", cwd).as_str()).unwrap();
    let sub_mat2: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/sq_mat2.csv", cwd).as_str()).unwrap();
    let out_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/add_mat.csv", cwd).as_str()).unwrap();

    let num_vars = log2i!(out_mat.len()) + log2i!(out_mat[0].len());
    let mut rng = Blake2b512Rng::setup();
    let g: Vec<_> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();

    let mut now = Instant::now();

    let inp_dim = (sub_mat1.len(), sub_mat1[0].len());

    let sub_mat1_mle = matrix_to_mle(sub_mat1, inp_dim, false).0;
    let sub_mat2_mle = matrix_to_mle(sub_mat2, inp_dim, false).0;
    let out_mle: DenseMultilinearExtension<Fp<MontBackend<FrConfig, 4>, 4>> = matrix_to_mle(out_mat, inp_dim, false).0;
    
    let eval = out_mle.fix_variables(g.as_slice())[0];

    let ones_mat = vec![vec![Fr::from(1); inp_dim.1]; inp_dim.0];
    let ones_mle: DenseMultilinearExtension<Fp<MontBackend<FrConfig, 4>, 4>> = matrix_to_mle(ones_mat, inp_dim, false).0;

    let mut circuit_mult = Vec::<(usize, usize, usize)>::new();
    
    let nr = inp_dim.0;
    let nc = inp_dim.1;
    let dim = log2i!(nr) + log2i!(nc);

    for i in 0..nr {
        for j in 0..nc {
            let ind = (i << log2i!(nc)) + j;
            circuit_mult.push((ind, ind, ind));
        }
    }

    let mut predicates_mult: Vec<(usize, Fr)> = Vec::new();

    for connec in circuit_mult {
        let ind  = rbin!((connec.2 << (2*dim)) + (connec.1 << dim) + connec.0, 3*dim);
        predicates_mult.push((ind, Fr::from(1)));
    }

    let preds_mle = SparseMultilinearExtension::from_evaluations(3*dim, &predicates_mult);
    
    
    let mut fs_rng_1 = Blake2b512Rng::setup();
    let mut fs_rng_2 = Blake2b512Rng::setup();

    let proof_1 = GKRRoundSumcheck::prove(&mut fs_rng_1, &preds_mle, &sub_mat1_mle, &ones_mle, &g);
    let proof_2 = GKRRoundSumcheck::prove(&mut fs_rng_2, &preds_mle, &ones_mle, &sub_mat2_mle, &g);


    let mut rng_1 = Blake2b512Rng::setup();
    let mut rng_2 = Blake2b512Rng::setup();

    let (subclaim_cos, eval1) = GKRRoundSumcheck::verify(&mut rng_1, sub_mat1_mle.num_vars, &proof_1, eval)
        .expect("Matrix add verification failed");

    let cos_mat_eval_u = sub_mat1_mle.evaluate(&(subclaim_cos.u));
    let cos_mat_eval_v = ones_mle.evaluate(&(subclaim_cos.v));

    let guv: Vec<_> = g
        .iter()
        .chain(subclaim_cos.u.iter())
        .chain(subclaim_cos.v.iter())
        .copied()
        .collect();
    let actual_evaluation = preds_mle.evaluate(&guv) * cos_mat_eval_u * cos_mat_eval_v;

    let result_cos = actual_evaluation == subclaim_cos.expected_evaluation;


    let (subclaim_sin, eval2) = GKRRoundSumcheck::verify(&mut rng_2, sub_mat2_mle.num_vars, &proof_2, eval)
    .expect("Matrix addition verification failed");

    let sin_mat_eval_u = ones_mle.evaluate(&(subclaim_sin.u));
    let sin_mat_eval_v = sub_mat2_mle.evaluate(&(subclaim_sin.v));

    let guv: Vec<_> = g
        .iter()
        .chain(subclaim_sin.u.iter())
        .chain(subclaim_sin.v.iter())
        .copied()
        .collect();
    let actual_evaluation = preds_mle.evaluate(&guv) * sin_mat_eval_u * sin_mat_eval_v;

    let result_sin = actual_evaluation == subclaim_sin.expected_evaluation;


    let tot_eval = eval1 + eval2;
    
    // println!("{}", tot_eval);
    // println!("{}", eval);

    assert!(tot_eval == eval, "Matrix addition verification failed");
    assert!(result_cos, "Matrix addition verification failed");
    assert!(result_sin, "Matrix addition verification failed");
}
