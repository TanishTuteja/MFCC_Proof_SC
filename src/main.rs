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

    let mut p_times: Vec<(&str, u128)> = Vec::new();
    let mut v_times: Vec<(&str, u128)> = Vec::new();




    // ===================   FILTERBANKS MATMUL   =====================================

    let features: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/features.csv", cwd).as_str()).unwrap();
    let pow_spec: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/pow_spec.csv", cwd).as_str()).unwrap();
    let filterbanks: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/filterbanks.csv", cwd).as_str()).unwrap();

    let mut elapsed_p: u128 = 0;
    let mut elapsed_v: u128 = 0;

    let mut now = Instant::now();
    let num_vars = log2i!(features.len()) + log2i!(features[0].len());
    let mut rng = Blake2b512Rng::setup();
    let g: Vec<_> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
    elapsed_v += now.elapsed().as_micros();

    now = Instant::now();

    let mut rng = ark_std::test_rng();
    let mut fs_rng = Blake2b512Rng::setup();

    let dim_output = (features.len(), features[0].len());
    let dim_left = (pow_spec.len(), pow_spec[0].len());
    let dim_right = (filterbanks.len(), filterbanks[0].len());

    let mut mm_prover = ProverMatMul::new(
        pow_spec.clone(),
        filterbanks, 
        dim_left, 
        dim_right, 
        Some(g.clone()),
        Some(&mut fs_rng),
    );


    let po = mm_prover.prove().unwrap();

    let features_mle = matrix_to_mle(features, dim_output, false).0;
    let features_mle_ds = DenseOrSparseMultilinearExtension::from(features_mle);
    let eval = features_mle_ds.fix_variables(g.as_slice())[0];

    elapsed_p += now.elapsed().as_micros();


    now = Instant::now();

    let mut fs_rng_v = Blake2b512Rng::setup();

    let eval_next = po.claimed_values.0;

    let mut mm_verifier = VerifierMatMul::new(
            // LayerInfoDense::,
            po,
            Some(eval),
            Some(g.clone()),
            Some(&mut fs_rng_v),
    );
    
    let _ = mm_verifier.verify().unwrap();
    
    elapsed_v += now.elapsed().as_micros();

    p_times.push(("Filterbanks matmul", elapsed_p));
    v_times.push(("Filterbanks matmul", elapsed_v));
    


    

    // ===================   ADDITION OF SQUARED (*1953)   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let mut g = g.split_at(log2i!(pow_spec.len())).0.to_vec();
    g.append(&mut mm_prover.input_randomness.unwrap());
    let eval = eval_next;

    let sub_mat1: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/sq_mat1.csv", cwd).as_str()).unwrap();
    let sub_mat2: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/sq_mat2.csv", cwd).as_str()).unwrap();

    now = Instant::now();

    let inp_dim = (sub_mat1.len(), sub_mat1[0].len());

    let sub_mat1_mle = matrix_to_mle(sub_mat1, inp_dim, false).0;
    let sub_mat2_mle = matrix_to_mle(sub_mat2, inp_dim, false).0;

    let prover_msg_sub_mat1 = sub_mat1_mle.evaluate(&g);
    let prover_msg_sub_mat2 = sub_mat2_mle.evaluate(&g);
    
    elapsed_p += now.elapsed().as_micros();

    now = Instant::now();
    let calc = Fr::from(1953) * (prover_msg_sub_mat1 + prover_msg_sub_mat2);
    let result = eval == calc;
    elapsed_v += now.elapsed().as_micros();
    
    assert!(result, "Verification for addition of squared cos and sin failed");

    p_times.push(("Addition of squared", elapsed_p));
    v_times.push(("Addition of squared", elapsed_v));
    





    // ===================   ELEMENTWISE SQUARING   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let eval_cos = prover_msg_sub_mat1;
    let eval_sin = prover_msg_sub_mat2;

    let cos_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/output_cos_257.csv", cwd).as_str()).unwrap();
    let sin_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/output_sin_257.csv", cwd).as_str()).unwrap();


    now = Instant::now();

    let mut circuit_mult = Vec::<(usize, usize, usize)>::new();
    
    let nr = cos_mat.len();
    let nc = cos_mat[0].len();
    let dim = log2i!(nr) + log2i!(nc);

    let inp_dim = (nr, nc);

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
    
    let elapsed_time = now.elapsed().as_micros();
    elapsed_p += elapsed_time;
    elapsed_v += elapsed_time;


    now = Instant::now();
    let cos_mat_mle = matrix_to_mle(cos_mat.clone(), inp_dim, false).0;
    let sin_mat_mle = matrix_to_mle(sin_mat.clone(), inp_dim, false).0;
    

    let mut fs_rng_1 = Blake2b512Rng::setup();
    let mut fs_rng_2 = Blake2b512Rng::setup();

    let proof_cos = GKRRoundSumcheck::prove(&mut fs_rng_1, &preds_mle, &cos_mat_mle, &cos_mat_mle, &g);
    let proof_sin = GKRRoundSumcheck::prove(&mut fs_rng_2, &preds_mle, &sin_mat_mle, &sin_mat_mle, &g);

    elapsed_p += now.elapsed().as_micros();
    

    now = Instant::now();
    let mut rng_1 = Blake2b512Rng::setup();
    let mut rng_2 = Blake2b512Rng::setup();

    let subclaim_cos = GKRRoundSumcheck::verify(&mut rng_1, cos_mat_mle.num_vars, &proof_cos, eval_cos)
        .expect("Matrix Squaring DFT cos verification failed");

    let cos_mat_eval_u = cos_mat_mle.evaluate(&(subclaim_cos.u));
    let cos_mat_eval_v = cos_mat_mle.evaluate(&(subclaim_cos.v));

    let guv: Vec<_> = g
        .iter()
        .chain(subclaim_cos.u.iter())
        .chain(subclaim_cos.v.iter())
        .copied()
        .collect();
    let actual_evaluation = preds_mle.evaluate(&guv) * cos_mat_eval_u * cos_mat_eval_v;

    let result_cos = actual_evaluation == subclaim_cos.expected_evaluation;


    let subclaim_sin = GKRRoundSumcheck::verify(&mut rng_2, sin_mat_mle.num_vars, &proof_sin, eval_sin)
    .expect("Matrix squaring DFT sin verification failed");

    let sin_mat_eval_u = sin_mat_mle.evaluate(&(subclaim_sin.u));
    let sin_mat_eval_v = sin_mat_mle.evaluate(&(subclaim_sin.v));

    let guv: Vec<_> = g
        .iter()
        .chain(subclaim_sin.u.iter())
        .chain(subclaim_sin.v.iter())
        .copied()
        .collect();
    let actual_evaluation = preds_mle.evaluate(&guv) * sin_mat_eval_u * sin_mat_eval_v;

    let result_sin = actual_evaluation == subclaim_sin.expected_evaluation;
    elapsed_v += now.elapsed().as_micros();

    assert!(result_cos, "DFT cos squaring verification failed");
    assert!(result_sin, "DFT sin squaring verification failed");

    p_times.push(("Elementwise Squaring", elapsed_p));
    v_times.push(("Elementwise Squaring", elapsed_v));
    




    // ===================   EXTRACT 257   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let g_cos = subclaim_cos.u;
    let g_sin = subclaim_sin.u;
    let cos_eval = cos_mat_eval_u;
    let sin_eval = sin_mat_eval_u;

    let cos_mat_full: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/output_cos.csv", cwd).as_str()).unwrap();
    let sin_mat_full: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/output_sin.csv", cwd).as_str()).unwrap();
    
    now = Instant::now();

    let mut circuit_mult = Vec::<(usize, usize)>::new();
    
    let nr = cos_mat_full.len();
    let nc = cos_mat_full[0].len();
    let dim = log2i!(nr) + log2i!(nc);
    
    let out_r = nr;
    let out_c = 257;
    let dim_out = log2i!(out_r) + log2i!(out_c);

    let inp_dim = (nr, nc);
    for inp_row in 0..nr {
        for inp_col in 0..257 {
            let ind = (inp_row << log2i!(nc))  + inp_col;
            let new_ind = (inp_row << log2i!(out_c)) + inp_col;
            circuit_mult.push((new_ind, ind));
        }
    }

    let mut predicates_mult: Vec<(usize, Fr)> = Vec::new();

    for connec in circuit_mult {
        let rev_index = rbin!(connec.0 * (1 << (dim)) + connec.1, dim + dim_out);
        predicates_mult.push((rev_index, Fr::from(1)));
    }

    let preds_mle = SparseMultilinearExtension::from_evaluations(dim + dim_out, &predicates_mult);
    
    let elapsed_time = now.elapsed().as_micros();
    elapsed_p += elapsed_time;
    elapsed_v += elapsed_time;


    now = Instant::now();

    let cos_full_mle = matrix_to_mle(cos_mat_full, inp_dim, false).0;
    let cos_full_mle_ds = DenseOrSparseMultilinearExtension::from(cos_full_mle);

    let preds_mle_fixed_cos = preds_mle.fix_variables(g_cos.as_slice());
    let preds_mle_fixed_cos_ds = DenseOrSparseMultilinearExtension::from(preds_mle_fixed_cos);

    let sin_full_mle = matrix_to_mle(sin_mat_full, inp_dim, false).0;
    let sin_full_mle_ds = DenseOrSparseMultilinearExtension::from(sin_full_mle);

    let preds_mle_fixed_sin = preds_mle.fix_variables(g_sin.as_slice());
    let preds_mle_fixed_sin_ds = DenseOrSparseMultilinearExtension::from(preds_mle_fixed_sin);

    let mut poly_cos = ListOfProductsOfPolynomials::new(cos_full_mle_ds.num_vars());
    let mut prod = Vec::new();
    prod.push(Rc::new(cos_full_mle_ds));
    prod.push(Rc::new(preds_mle_fixed_cos_ds));
    poly_cos.add_product(prod, Fr::from(1 as u32));
    let mut fs_rng_cos = Blake2b512Rng::setup();
    fs_rng_cos.feed(&poly_cos.info()).unwrap();

    let mut poly_sin = ListOfProductsOfPolynomials::new(sin_full_mle_ds.num_vars());
    let mut prod = Vec::new();
    prod.push(Rc::new(sin_full_mle_ds));
    prod.push(Rc::new(preds_mle_fixed_sin_ds));
    poly_sin.add_product(prod, Fr::from(1 as u32));
    let mut fs_rng_sin = Blake2b512Rng::setup();
    fs_rng_sin.feed(&poly_sin.info()).unwrap();

    let (prover_msgs_cos, mut prover_state_cos) = MLSC::prove(
        &poly_cos, 
        &mut fs_rng_cos
    ).unwrap();
    let oracle_randomness_cos = &[Fr::rand(&mut fs_rng_cos)];
    prover_state_cos.randomness.push(oracle_randomness_cos[0]);
    fs_rng_cos.feed(&oracle_randomness_cos[0]).unwrap();

    let proof_cos = simpgkr_final_access(&prover_state_cos, oracle_randomness_cos);
    

    let (prover_msgs_sin, mut prover_state_sin) = MLSC::prove(
        &poly_sin, 
        &mut fs_rng_sin
    ).unwrap();
    let oracle_randomness_sin = &[Fr::rand(&mut fs_rng_sin)];
    prover_state_sin.randomness.push(oracle_randomness_sin[0]);
    fs_rng_sin.feed(&oracle_randomness_sin[0]).unwrap();

    let proof_sin = simpgkr_final_access(&prover_state_sin, oracle_randomness_sin);
    
    elapsed_p += now.elapsed().as_micros();


    now = Instant::now();

    let mut fs_rng_cos = Blake2b512Rng::setup();
    let mut layer_verifier_cos = VerifierConv2D::<Fr>::new(
        cos_eval,
        &mut fs_rng_cos,
    );

    let verifier_messages_cos = layer_verifier_cos.verify(
        &poly_cos.info(),
        proof_cos,
        &prover_msgs_cos
    ).unwrap();


    let mut fs_rng_sin = Blake2b512Rng::setup();
    let mut layer_verifier_sin = VerifierConv2D::<Fr>::new(
        sin_eval,
        &mut fs_rng_sin,
    );

    let verifier_messages_sin = layer_verifier_sin.verify(
        &poly_sin.info(),
        proof_sin,
        &prover_msgs_sin
    ).unwrap();

    elapsed_v += now.elapsed().as_micros();

    p_times.push(("Extracting 257 from 512 DFT", elapsed_p));
    v_times.push(("Extracting 257 from 512 DFT", elapsed_v));
    




    // ===================   DFT MATMUL   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let g_cos = prover_state_cos.randomness;
    let g_sin = prover_state_sin.randomness;
    let cos_eval = proof_cos.0;
    let sin_eval = proof_sin.0;

    let windowed_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/framed_aud.csv", cwd).as_str()).unwrap();
    let dft_cos_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/cos_vals.csv", cwd).as_str()).unwrap();
    let dft_sin_mat: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/sin_vals.csv", cwd).as_str()).unwrap();

    
    let now = Instant::now();

    let mut rng = ark_std::test_rng();
    let mut fs_rng_1 = Blake2b512Rng::setup();
    let mut fs_rng_2 = Blake2b512Rng::setup();

    let dim_output = (windowed_mat.len(), dft_cos_mat[0].len());
    let dim_left = (windowed_mat.len(), windowed_mat[0].len());
    let dim_right = (dft_cos_mat.len(), dft_cos_mat[0].len());

    let mut mm_prover_cos = ProverMatMul::new(
        windowed_mat.clone(),
        dft_cos_mat, 
        dim_left, 
        dim_right, 
        Some(g_cos.clone()),
        Some(&mut fs_rng_1),
    );

    let mut mm_prover_sin = ProverMatMul::new(
        windowed_mat.clone(),
        dft_sin_mat, 
        dim_left, 
        dim_right, 
        Some(g_sin.clone()),
        Some(&mut fs_rng_2),
    );

    let po_cos = mm_prover_cos.prove().unwrap();
    let po_sin = mm_prover_sin.prove().unwrap();
    
    elapsed_p += now.elapsed().as_micros();


    let now = Instant::now();

    let mut fs_rng_v_1 = Blake2b512Rng::setup();
    let mut fs_rng_v_2 = Blake2b512Rng::setup();

    let cos_eval_next = po_cos.claimed_values.0;
    let sin_eval_next = po_sin.claimed_values.0;

    let mut mm_verifier_cos = VerifierMatMul::new(
            // LayerInfoDense::,
            po_cos,
            Some(cos_eval),
            Some(g_cos.clone()),
            Some(&mut fs_rng_v_1),
    );
    
    let mut mm_verifier_sin = VerifierMatMul::new(
        // LayerInfoDense::,
        po_sin,
        Some(sin_eval),
        Some(g_sin.clone()),
        Some(&mut fs_rng_v_2),
    );
    
    let _ = mm_verifier_cos.verify().unwrap();
    let _ = mm_verifier_sin.verify().unwrap();
    
    elapsed_v += now.elapsed().as_micros();

    p_times.push(("DFT mat mul", elapsed_p));
    v_times.push(("DFT mat mul", elapsed_v));
    




    // ===================   FRAMING   =====================================

    elapsed_p = 0;
    elapsed_v = 0;


    let mut g_cos = g_cos.split_at(log2i!(windowed_mat.len())).0.to_vec();
    g_cos.append(&mut mm_prover_cos.input_randomness.unwrap());
    let mut g_sin = g_sin.split_at(log2i!(windowed_mat.len())).0.to_vec();
    g_sin.append(&mut mm_prover_sin.input_randomness.unwrap());
    let cos_eval = cos_eval_next;
    let sin_eval = sin_eval_next;

    let mut rng = Blake2b512Rng::setup();
    let alpha = Fr::rand(&mut rng);
    let beta = Fr::rand(&mut rng);

    let eval = alpha * cos_eval + beta * sin_eval;

    let emph_audio: Vec<Vec<Fp<MontBackend<FrConfig, 4>, 4>>> = read_matrix(format!("{}/data/emp_audio.csv", cwd).as_str()).unwrap();
    

    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut circuit_mult = Vec::<(usize, usize)>::new();
    
    let nr = emph_audio.len();
    let dim = log2i!(nr);
    
    let fs = 512;
    let stride = 256;

    let out_c = fs;

    let inp_dim = (nr, 1);
    let mut inp_row = 0;
    let mut out_row = 0;
    while (inp_row + fs - 1) < nr {
        for j in 0..fs {
            let ind = inp_row  + j;
            let new_ind = (out_row << log2i!(out_c)) + j;
            circuit_mult.push((new_ind, ind));
        }
        inp_row += stride;
        out_row += 1;
    }

    let out_r = out_row;
    let dim_out = log2i!(out_r) + log2i!(out_c);

    let mut predicates_mult_cos: Vec<(usize, Fr)> = Vec::new();
    let mut predicates_mult_sin: Vec<(usize, Fr)> = Vec::new();

    for connec in circuit_mult {
        let rev_index = rbin!(connec.0 * (1 << (dim)) + connec.1, dim + dim_out);
        predicates_mult_cos.push((rev_index, alpha));
        predicates_mult_sin.push((rev_index, beta));
    }

    let preds_mle_cos = SparseMultilinearExtension::from_evaluations(dim + dim_out, &predicates_mult_cos);
    let preds_mle_sin = SparseMultilinearExtension::from_evaluations(dim + dim_out, &predicates_mult_sin);

    let preds_mle_fixed_cos = preds_mle_cos.fix_variables(g_cos.as_slice());
    let preds_mle_fixed_sin = preds_mle_sin.fix_variables(g_sin.as_slice());
    let preds_mle_fixed = preds_mle_fixed_cos + preds_mle_fixed_sin;
    let preds_mle_fixed_ds = DenseOrSparseMultilinearExtension::from(preds_mle_fixed);
    
    let elapsed_time = now.elapsed().as_micros();
    elapsed_p += elapsed_time;
    elapsed_v += elapsed_time;

    let now = Instant::now();

    let emph_audio_mle = matrix_to_mle(emph_audio.clone(), inp_dim, false).0;
    let emph_audio_mle_ds = DenseOrSparseMultilinearExtension::from(emph_audio_mle);

    let mut poly = ListOfProductsOfPolynomials::new(emph_audio_mle_ds.num_vars());
    let mut prod: Vec<Rc<DenseOrSparseMultilinearExtension<Fp<MontBackend<FrConfig, 4>, 4>>>> = Vec::new();
    prod.push(Rc::new(emph_audio_mle_ds));
    prod.push(Rc::new(preds_mle_fixed_ds));
    poly.add_product(prod, Fr::from(1 as u32));

    fs_rng.feed(&poly.info()).unwrap();
    
    let (prover_msgs_framing, mut prover_state_framing) = MLSC::prove(
        &poly, 
        &mut fs_rng
    ).unwrap();

    let oracle_randomness = &[Fr::rand(&mut fs_rng)];

    prover_state_framing.randomness.push(oracle_randomness[0]);
    fs_rng.feed(&oracle_randomness[0]).unwrap();

    let proof_framing = simpgkr_final_access(&prover_state_framing, oracle_randomness);
    
    elapsed_p += now.elapsed().as_micros();


    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut layer_verifier_framing = VerifierConv2D::<Fr>::new(
        // prover_output_conv.proof.0,
        eval,
        &mut fs_rng,
    );

    let verifier_messages_framing = layer_verifier_framing.verify(
        &poly.info(),
        proof_framing,
        &prover_msgs_framing
    ).unwrap();

    elapsed_v += now.elapsed().as_micros();

    p_times.push(("Framing", elapsed_p));
    v_times.push(("Framing", elapsed_v));
    




    // ===================   PADDING   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let g: Vec<Fp<MontBackend<FrConfig, 4>, 4>> = prover_state_framing.randomness;
    let eval = proof_framing.0;

    let emph_aud_unp = read_matrix(format!("{}/data/unp_emph_audio.csv", cwd).as_str()).unwrap();

    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut circuit_mult = Vec::<(usize, usize)>::new();
    
    let nr = emph_aud_unp.len();
    let dim = log2i!(nr);

    let out_r = emph_audio.len();
    let out_c = 1;
    let dim_out = log2i!(out_r);

    let inp_dim = (nr, 1);
    for inp_row in  0..nr {
        let ind = inp_row;
        let new_ind = inp_row;
        circuit_mult.push((new_ind, ind));
    }

    let mut predicates_mult: Vec<(usize, Fr)> = Vec::new();

    for connec in circuit_mult {
        let rev_index = rbin!(connec.0 * (1 << (dim)) + connec.1, dim + dim_out);
        predicates_mult.push((rev_index, Fr::from(1)));
    }

    let preds_mle = SparseMultilinearExtension::from_evaluations(dim + dim_out, &predicates_mult);
    
    let preds_mle_fixed = preds_mle.fix_variables(g.as_slice());
    let preds_mle_fixed_ds = DenseOrSparseMultilinearExtension::from(preds_mle_fixed);

    let elapsed_time = now.elapsed().as_micros();
    elapsed_p += elapsed_time;
    elapsed_v += elapsed_time;


    let now = Instant::now();

    let emph_aud_unp_mle = matrix_to_mle(emph_aud_unp, (nr, 1), false).0;
    let emph_aud_unp_mle_ds = DenseOrSparseMultilinearExtension::from(emph_aud_unp_mle);

    let mut poly = ListOfProductsOfPolynomials::new(emph_aud_unp_mle_ds.num_vars());
    let mut prod: Vec<Rc<DenseOrSparseMultilinearExtension<Fp<MontBackend<FrConfig, 4>, 4>>>> = Vec::new();
    prod.push(Rc::new(emph_aud_unp_mle_ds));
    prod.push(Rc::new(preds_mle_fixed_ds));
    poly.add_product(prod, Fr::from(1 as u32));

    fs_rng.feed(&poly.info()).unwrap();
    
    let (prover_msgs_padding, mut prover_state_padding) = MLSC::prove(
        &poly, 
        &mut fs_rng
    ).unwrap();

    let oracle_randomness = &[Fr::rand(&mut fs_rng)];

    prover_state_padding.randomness.push(oracle_randomness[0]);
    fs_rng.feed(&oracle_randomness[0]).unwrap();

    let proof_padding = simpgkr_final_access(&prover_state_padding, oracle_randomness);
    
    elapsed_p += now.elapsed().as_micros();


    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut layer_verifier_padding = VerifierConv2D::<Fr>::new(
        // prover_output_conv.proof.0,
        eval,
        &mut fs_rng,
    );

    let verifier_messages_padding = layer_verifier_padding.verify(
        &poly.info(),
        proof_padding,
        &prover_msgs_padding
    ).unwrap();

    elapsed_v += now.elapsed().as_micros();

    p_times.push(("Padding", elapsed_p));
    v_times.push(("Padding", elapsed_v));





    // ===================   PRE-EMPH STEP 2 (100*a[i] - 97*b[i])   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let g: Vec<Fp<MontBackend<FrConfig, 4>, 4>> = prover_state_padding.randomness;
    let eval = proof_padding.0;

    let shifted_audio = read_matrix(format!("{}/data/shifted_audio.csv", cwd).as_str()).unwrap();
    let raw_audio = read_matrix(format!("{}/data/raw_aud.csv", cwd).as_str()).unwrap();
    
    let now = Instant::now();

    let nr = raw_audio.len();

    let inp_dim = (shifted_audio.len(), shifted_audio[0].len());
    let shifted_audio_mle = matrix_to_mle(shifted_audio, inp_dim, false).0;
    let raw_audio_mle = matrix_to_mle(raw_audio, inp_dim, false).0;

    let prover_msg_raw_aud = raw_audio_mle.evaluate(&g);
    let prover_msg_shifted_aud = shifted_audio_mle.evaluate(&g);

    elapsed_p += now.elapsed().as_micros();

    let now = Instant::now();
    let calc = Fr::from(100) * prover_msg_raw_aud - Fr::from(97) * prover_msg_shifted_aud;
    let result = eval == calc;
    elapsed_v += now.elapsed().as_micros();

    assert!(result, "Verification for pre-emph step 2 failed");

    let now = Instant::now();
    let result = raw_audio_mle.evaluate(&g) == prover_msg_raw_aud;
    elapsed_v += now.elapsed().as_micros();

    assert!(result, "Input verification failed pre-emph step 2");

    p_times.push(("Pre Emphasis step 2: subtraction", elapsed_p));
    v_times.push(("Pre Emphasis step 2: subtraction", elapsed_v));
    


    
    
    // ===================   PRE-EMPH STEP 1 (SHIFTING)   =====================================

    elapsed_p = 0;
    elapsed_v = 0;

    let eval = prover_msg_shifted_aud;


    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut circuit_mult = Vec::<(usize, usize)>::new();
    
    let dim = log2i!(nr);

    let out_r = nr;
    let out_c = 1;
    let dim_out = log2i!(out_r);

    let inp_dim = (nr, 1);
    for inp_row in  0..nr-1 {
        let ind = inp_row;
        let new_ind = inp_row + 1;
        circuit_mult.push((new_ind, ind));
    }

    let mut predicates_mult: Vec<(usize, Fr)> = Vec::new();

    for connec in circuit_mult {
        let rev_index = rbin!(connec.0 * (1 << (dim)) + connec.1, dim + dim_out);
        predicates_mult.push((rev_index, Fr::from(1)));
    }

    let preds_mle = SparseMultilinearExtension::from_evaluations(dim + dim_out, &predicates_mult);
    
    let preds_mle_fixed = preds_mle.fix_variables(g.as_slice());
    let preds_mle_fixed_ds = DenseOrSparseMultilinearExtension::from(preds_mle_fixed);

    let elapsed_time = now.elapsed().as_micros();
    elapsed_p += elapsed_time;
    elapsed_v += elapsed_time;


    let now = Instant::now();

    let raw_audio_mle_ds = DenseOrSparseMultilinearExtension::from(raw_audio_mle.clone());

    let mut poly = ListOfProductsOfPolynomials::new(raw_audio_mle_ds.num_vars());
    let mut prod: Vec<Rc<DenseOrSparseMultilinearExtension<Fp<MontBackend<FrConfig, 4>, 4>>>> = Vec::new();
    prod.push(Rc::new(raw_audio_mle_ds));
    prod.push(Rc::new(preds_mle_fixed_ds));
    poly.add_product(prod, Fr::from(1 as u32));

    fs_rng.feed(&poly.info()).unwrap();
    
    let (prover_msgs_shifting, mut prover_state_shifting) = MLSC::prove(
        &poly, 
        &mut fs_rng
    ).unwrap();

    let oracle_randomness = &[Fr::rand(&mut fs_rng)];

    prover_state_shifting.randomness.push(oracle_randomness[0]);
    fs_rng.feed(&oracle_randomness[0]).unwrap();

    let proof_shifting = simpgkr_final_access(&prover_state_shifting, oracle_randomness);
    
    elapsed_p += now.elapsed().as_micros();


    let now = Instant::now();

    let mut fs_rng = Blake2b512Rng::setup();

    let mut layer_verifier_shifting = VerifierConv2D::<Fr>::new(
        // prover_output_conv.proof.0,
        eval,
        &mut fs_rng,
    );

    let verifier_messages_shifting = layer_verifier_shifting.verify(
        &poly.info(),
        proof_shifting,
        &prover_msgs_shifting
    ).unwrap();

    elapsed_v += now.elapsed().as_micros();


    let g: Vec<Fp<MontBackend<FrConfig, 4>, 4>> = prover_state_shifting.randomness;
    let eval = proof_shifting.0;

    let now = Instant::now();
    let result = raw_audio_mle.evaluate(&g) == eval;
    elapsed_v += now.elapsed().as_micros();

    assert!(result, "Input verification failed");

    p_times.push(("Pre Emphasis step 1: shifting", elapsed_p));
    v_times.push(("Pre Emphasis step 1: shifting", elapsed_v));
    
    



    // ===================   PRINT TIMES   =====================================

    let mut total_p: u128 = 0;
    for i in 0..p_times.len(){
        println!("Prove {} - {}", p_times[i].0, p_times[i].1);
        total_p += p_times[i].1;
    }

    println!();

    let mut total_v: u128 = 0;
    for i in 0..v_times.len(){
        println!("Verify {} - {}", v_times[i].0, v_times[i].1);
        total_v += v_times[i].1;
    }

    print!("\n\n");

    println!("Total Prover Time - {}", total_p);
    println!("Total Verifier Time - {}", total_v);
    
}

// Function used by simple gkr (one input) prover at the end to evaluate inputs
pub fn simpgkr_final_access(prover_state: &ProverState<Fr>, final_randomness: &[Fr]) -> (Fr,Fr) 
{
        let mut input_mle = prover_state.flattened_ml_extensions[0].clone();
        let mut kernel_mle = prover_state.flattened_ml_extensions[1].clone();

        input_mle = input_mle.fix_variables(final_randomness);
        kernel_mle = kernel_mle.fix_variables(final_randomness);

        // Checking all variables were fixed
        assert_eq!(input_mle.num_vars(), 0);
        assert_eq!(kernel_mle.num_vars(), 0);

        let input_mle_eval = input_mle.evaluations()[0];
        let kernel_mle_eval = kernel_mle.evaluations()[0];

        (input_mle_eval, kernel_mle_eval)

}