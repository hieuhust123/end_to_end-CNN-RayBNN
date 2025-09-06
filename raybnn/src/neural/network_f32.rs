extern crate arrayfire;
use arrayfire::af_print;
use std::collections::HashMap;
use nohash_hasher;

use crate::neural::activation_f32::UAF;
use crate::neural::activation_f32::deriUAF;


use crate::graph::large_sparse_i32::COO_batch_find;

use serde::{Serialize, Deserialize};
use crate::graph::tree_i32::find_unique;


const COO_find_limit: u64 = 1500000000;


const one: f32 = 1.0;


#[derive(Serialize, Deserialize)]
pub struct network_metadata_type {
    pub neuron_size: u64,
    pub input_size: u64,
	pub output_size: u64,
	pub proc_num: u64,
	pub active_size: u64,
	pub space_dims: u64,
	pub step_num: u64,
    pub batch_size: u64,
	pub del_unused_neuron: bool,
	pub time_step: f32,
	pub nratio: f32,
	pub neuron_std: f32,
	pub sphere_rad: f32,
	pub neuron_rad: f32,
	pub con_rad: f32,
    pub init_prob: f32,
    pub add_neuron_rate: f32,
    pub del_neuron_rate: f32,
	pub center_const: f32,
	pub spring_const: f32,
	pub repel_const: f32
}





#[derive(Serialize, Deserialize)]
pub struct neural_network_type {
    pub netdata: network_metadata_type,
    pub WRowIdxCSR: arrayfire::Array<i32>,
    pub WColIdx: arrayfire::Array<i32>,
    pub network_params: arrayfire::Array<f32>,
    pub glia_pos: arrayfire::Array<f32>,
    pub neuron_pos: arrayfire::Array<f32>,
    pub neuron_idx: arrayfire::Array<i32>
}








pub fn print_netdata(
    netdata: &network_metadata_type
)
{
    println!("\n\n******Network Information******");
    println!("neuron_size: {}",netdata.neuron_size);
    println!("input_size: {}",netdata.input_size);
    println!("output_size: {}",netdata.output_size);
    println!("proc_num: {}",netdata.proc_num);
    println!("active_size: {}",netdata.active_size);
    println!("space_dims: {}",netdata.space_dims);
    println!("step_num: {}",netdata.step_num);
    println!("batch_size: {}",netdata.batch_size);
    println!("del_unused_neuron: {}",netdata.del_unused_neuron);



    println!("time_step: {}",netdata.time_step);
    println!("nratio: {}",netdata.nratio);
    println!("neuron_std: {}",netdata.neuron_std);
    println!("sphere_rad: {}",netdata.sphere_rad);
    println!("neuron_rad: {}",netdata.neuron_rad);
    println!("con_rad: {}",netdata.con_rad);
    println!("init_prob: {}",netdata.init_prob);
    println!("add_neuron_rate: {}",netdata.add_neuron_rate);
    println!("del_neuron_rate: {}",netdata.del_neuron_rate);
    println!("center_const: {}",netdata.center_const);
    println!("spring_const: {}",netdata.spring_const);
    println!("repel_const: {}",netdata.repel_const);
    println!("************\n\n");
}














pub fn clone_netdata(netdata: &network_metadata_type) -> network_metadata_type
{

	let neuron_size: u64 = netdata.neuron_size.clone();
	let input_size: u64 = netdata.input_size.clone();
	let output_size: u64 = netdata.output_size.clone();
	let proc_num: u64 = netdata.proc_num.clone();
	let active_size: u64 = netdata.active_size.clone();
	let space_dims: u64 = netdata.space_dims.clone();
	let step_num: u64 = netdata.step_num.clone();
	let batch_size: u64 = netdata.batch_size.clone();


	let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


	let time_step: f32 = netdata.time_step.clone();
	let nratio: f32 = netdata.nratio.clone();
	let neuron_std: f32 = netdata.neuron_std.clone();
	let sphere_rad: f32 = netdata.sphere_rad.clone();
	let neuron_rad: f32 = netdata.neuron_rad.clone();
	let con_rad: f32 = netdata.con_rad.clone();
	let init_prob: f32 = netdata.init_prob.clone();
	let add_neuron_rate: f32 = netdata.add_neuron_rate.clone();
	let del_neuron_rate: f32 = netdata.del_neuron_rate.clone();
	let center_const: f32 = netdata.center_const.clone();
	let spring_const: f32 = netdata.spring_const.clone();
	let repel_const: f32 = netdata.repel_const.clone();


	let newnetdata = network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: step_num,
		batch_size: batch_size,
		del_unused_neuron: del_unused_neuron,

		time_step: time_step,
		nratio: nratio,
		neuron_std: neuron_std,
		sphere_rad: sphere_rad,
		neuron_rad: neuron_rad,
		con_rad: con_rad,
		init_prob: init_prob,
		add_neuron_rate: add_neuron_rate,
		del_neuron_rate: del_neuron_rate,
		center_const: center_const,
		spring_const: spring_const,
		repel_const: repel_const
	};



    newnetdata
}





pub fn clone_neural_network(neural_network: &neural_network_type) -> neural_network_type
{


    let mut new_neural_network =    neural_network_type {
        netdata: clone_netdata(&neural_network.netdata),
        WRowIdxCSR: neural_network.WRowIdxCSR.clone(),
        WColIdx: neural_network.WColIdx.clone(),
        network_params: neural_network.network_params.clone(),
        glia_pos: neural_network.glia_pos.clone(),
        neuron_pos: neural_network.neuron_pos.clone(),
        neuron_idx: neural_network.neuron_idx.clone()
    };
    
    
    new_neural_network
}








pub fn create_nullnetdata() -> network_metadata_type
{

	let netdata: network_metadata_type = network_metadata_type {
		neuron_size: 0,
	    input_size: 0,
		output_size: 0,
		proc_num: 0,
		active_size: 0,
		space_dims: 0,
		step_num: 0,
		batch_size: 0,
		del_unused_neuron: false,

		time_step: 0.0,
		nratio: 0.0,
		neuron_std: 0.0,
		sphere_rad: 0.0,
		neuron_rad: 0.0,
		con_rad: 0.0,
		init_prob: 0.0,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.0,
		spring_const: 0.0,
		repel_const: 0.0
	};



    netdata
}












pub fn UAF_initial_as_identity(
		netdata: &network_metadata_type,
		//H: &mut arrayfire::Array<f32>,
		A: &mut arrayfire::Array<f32>,
		B: &mut arrayfire::Array<f32>,
		C: &mut arrayfire::Array<f32>,
		D: &mut arrayfire::Array<f32>,
		E: &mut arrayfire::Array<f32>
	)
    {
        let neuron_size: u64 = netdata.neuron_size.clone();
		let input_size: u64 = netdata.input_size.clone();
		let output_size: u64 = netdata.output_size.clone();
		let proc_num: u64 = netdata.proc_num.clone();
		let active_size: u64 = netdata.active_size.clone();
		let space_dims: u64 = netdata.space_dims.clone();
		let step_num: u64 = netdata.step_num.clone();


		let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


		let time_step: f32 = netdata.time_step.clone();
		let nratio: f32 = netdata.nratio.clone();
		let neuron_std: f32 = netdata.neuron_std.clone();
		let sphere_rad: f32 = netdata.sphere_rad.clone();
		let neuron_rad: f32 = netdata.neuron_rad.clone();
		let con_rad: f32 = netdata.con_rad.clone();
		let center_const: f32 = netdata.center_const.clone();
		let spring_const: f32 = netdata.spring_const.clone();
		let repel_const: f32 = netdata.repel_const.clone();





		let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
        //  *H = 0.001*neuron_std*arrayfire::randn::<f32>(H_dims);
        *A = one + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*B = 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*C = 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*D = -one + 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);
		*E = 0.00001*neuron_std*arrayfire::randn::<f32>(H_dims);


}








pub fn UAF_initial_as_tanh(
    netdata: &network_metadata_type,
    //H: &mut arrayfire::Array<f32>,
    A: &mut arrayfire::Array<f32>,
    B: &mut arrayfire::Array<f32>,
    C: &mut arrayfire::Array<f32>,
    D: &mut arrayfire::Array<f32>,
    E: &mut arrayfire::Array<f32>
)
{
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();
    let active_size: u64 = netdata.active_size.clone();
    let space_dims: u64 = netdata.space_dims.clone();
    let step_num: u64 = netdata.step_num.clone();


    let del_unused_neuron: bool = netdata.del_unused_neuron.clone();


    let time_step: f32 = netdata.time_step.clone();
    let nratio: f32 = netdata.nratio.clone();
    let neuron_std: f32 = netdata.neuron_std.clone();
    let sphere_rad: f32 = netdata.sphere_rad.clone();
    let neuron_rad: f32 = netdata.neuron_rad.clone();
    let con_rad: f32 = netdata.con_rad.clone();
    let center_const: f32 = netdata.center_const.clone();
    let spring_const: f32 = netdata.spring_const.clone();
    let repel_const: f32 = netdata.repel_const.clone();





    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    //   *H = 0.0000001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *A = 2.12616013f32 + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *B = (1.0f32/2.12616013f32) + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *C = 0.0000001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *D = 2.12616013f32 + 0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);
    *E = -1.0f32 +  0.0001*neuron_std*arrayfire::randn::<f32>(H_dims);


}








pub fn xavier_init(
    in_idx: &arrayfire::Array<i32>,
    WRowIdxCOO: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,
    neuron_size: u64,
    depth: u64,

    WValues: &mut arrayfire::Array<f32>,
    H: &mut arrayfire::Array<f32>,
)
{
    *WValues = 0.000001f32*arrayfire::randn::<f32>(WValues.dims());


    let H_dims = arrayfire::Dim4::new(&[neuron_size,1,1,1]);
    *H = 0.000001f32*arrayfire::randn::<f32>(H_dims);



    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut out_idx = in_idx.clone();
    let mut valsel = arrayfire::constant::<i32>(0,temp_dims);

    let COO_batch_size = 1 + ((COO_find_limit/WColIdx.dims()[0]) as u64);


    let mut input_degree = 0;
    let mut output_degree = 0;

    for i in 0..depth
    {
        valsel = COO_batch_find(WColIdx, &out_idx, COO_batch_size);
        if valsel.dims()[0] == 0
        {
            break;
        }
        out_idx = arrayfire::lookup(WRowIdxCOO, &valsel, 0);

        out_idx = find_unique(&out_idx, neuron_size);


        output_degree = valsel.dims()[0];

        let mulitiplier = (6.0f32/((input_degree + output_degree) as f32) ).sqrt()*2.0f32;
        let mut newWValues = arrayfire::randu::<f32>(valsel.dims());
        newWValues = (newWValues - 0.5f32)*( mulitiplier );

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel, 0, None);
        arrayfire::assign_gen(WValues, &idxrs, &newWValues);

        drop(newWValues);
        drop(idxrs);


        let mut newH = arrayfire::randu::<f32>(out_idx.dims());
        newH = (newH - 0.5f32)*( mulitiplier );

        let mut idxrs2 = arrayfire::Indexer::default();
        idxrs2.set_index(&out_idx, 0, None);
        arrayfire::assign_gen(H, &idxrs2, &newH);



        if out_idx.dims()[0] == 0
        {
            break;
        }

        input_degree = valsel.dims()[0];
    }


}











/*
Forward pass using CSR weighted adjacency sparse matrices and UAF. 
Generates all internal states and the neural network output


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
WRowIdxCSR:          Row sparse matrix of the weighted adjacency matrix
WColIdx:             Column sparse matrix of the weighted adjacency matrix
Wseqs:               Indexes of the weight parameters
Hseqs:               Indexes of the bias parameters
Aseqs:               Indexes of the UAF A parameters
Bseqs:               Indexes of the UAF B parameters
Cseqs:               Indexes of the UAF C parameters
Dseqs:               Indexes of the UAF D parameters
Eseqs:               Indexes of the UAF E parameters
network_params:      All trainable parameters in the network


Outputs:
Z:                   Internal State Matrix Z
Q:                   Internal State Matrix Q

*/

pub fn state_space_forward_batch(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f32>,
    
    WRowIdxCSR: &arrayfire::Array<i32>,
    WColIdx: &arrayfire::Array<i32>,


    Wseqs: &[arrayfire::Seq<i32>; 1],
    Hseqs: &[arrayfire::Seq<i32>; 1],
    Aseqs: &[arrayfire::Seq<i32>; 1],
    Bseqs: &[arrayfire::Seq<i32>; 1],
    Cseqs: &[arrayfire::Seq<i32>; 1],
    Dseqs: &[arrayfire::Seq<i32>; 1],
    Eseqs: &[arrayfire::Seq<i32>; 1],
    network_params: &arrayfire::Array<f32>,



    Z: &mut arrayfire::Array<f32>,
    Q: &mut arrayfire::Array<f32>
) {
    //println!("START forward pass! ");
    let neuron_size: u64 = netdata.neuron_size.clone();
    //let proc_num: u64 = netdata.proc_num.clone();
    let input_size: u64 = netdata.input_size.clone();
    let batch_size: u64 = netdata.batch_size.clone();

    let Zslices:i64 = Z.dims()[2] as i64;

    let X_slices:i64 = X.dims()[2] as i64;
    //println!("X_slices: {:?} \n", X_slices);
    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    let mut S = arrayfire::constant::<f32>(0.0, S_dims);
    //println!("S: {:?} \n", S);
    let mut tempx =  arrayfire::slice(X, 0);
    //af_print!("tempx: ", tempx);
    let seqs = &[arrayfire::Seq::new(0.0f32, (input_size-1) as f32, 1.0f32),  arrayfire::Seq::default()];
    //println!("seqs: {:?} \n", seqs);




    let H = arrayfire::index(&network_params, Hseqs);
    let A = arrayfire::index(&network_params, Aseqs);
    let B = arrayfire::index(&network_params, Bseqs);
    let C = arrayfire::index(&network_params, Cseqs);
    let D = arrayfire::index(&network_params, Dseqs);
    let E = arrayfire::index(&network_params, Eseqs);


    let WValues = arrayfire::index(network_params, Wseqs);
    // println!("network_params.dims(): {:?}", network_params.dims());
    // println!("Wseqs: {:?}", Wseqs);

    // println!("WValues.dims(): {:?}", WValues.dims());
    // println!("WRowIdxCSR.dims(): {:?}", WRowIdxCSR.dims());
    // println!("WColIdx.dims(): {:?}", WColIdx.dims());
    // println!("neuron_size: {}", neuron_size);

// Check if dimensions match
if WValues.dims()[0] != WColIdx.dims()[0] {
    panic!("WValues and WColIdx size mismatch");
}


    let W = arrayfire::sparse::<f32>(
        neuron_size,
        neuron_size,
        &WValues,
        WRowIdxCSR,
        WColIdx,
        arrayfire::SparseFormat::CSR
    );
    //println!("W: {:?} \n",W);
    //af_print!("W: ", W);
    for i in 0i64..Zslices
    {
        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            //af_print!("tempx: ", tempx);
            arrayfire::assign_seq(&mut S, seqs, &tempx);
            //af_print!("S: ", S);
            drop(tempx);
        }
        else
        {
            arrayfire::assign_seq(&mut S, seqs, &X);
            //af_print!("S: ", S);
        }
        

        S = arrayfire::matmul(&W, &S, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE);
        S = arrayfire::add(&S, &H, true);
        arrayfire::set_slice(Z, &S, i);

        S = UAF(&S,&A,&B,&C,&D,&E);
        arrayfire::set_slice(Q, &S, i);
    }

    //println!("FINISH forward pass! \n");
    
}





















/*
Backward pass using CSR weighted adjacency sparse matrices and UAF.
Generates the gradients of the sparse weighted adjacency matrix


Inputs
netdata:             Neural Network Metadata
X:                   Input Matrix
network_params:      All trainable parameters in the network
Z:                   Internal State matrix Z
Q:                   Internal State matrix Q
Y:                   Target Matrix to fit
loss_grad:           Loss function
neuron_idx:          Indexes of the neurons
idxsel_out:          Indexes of the UAF parameters
valsel_out:          Indexes of the UAF values
cvec_out:            Indexes of the UAF column sparse vector
dXsel_out:           Indexes of the dX values
nrows_out:           Number of rows in UAF
sparseval_out:       Indexes of the values in the sparse matrix
sparserow_out:       Indexes of the rows in the sparse matrix
sparsecol_out:       Indexes of the columns in the sparse matrix
Hidxsel_out:         Indexes of the bias vector H
Aidxsel_out:         Indexes of the UAF vector A
Bidxsel_out:         Indexes of the UAF vector B
Cidxsel_out:         Indexes of the UAF vector C
Didxsel_out:         Indexes of the UAF vector D
Eidxsel_out:         Indexes of the UAF vector E
combidxsel_out:      Indexes of the all vectors
dAseqs_out:          Indexes of the dA
dBseqs_out:          Indexes of the dB
dCseqs_out:          Indexes of the dC
dDseqs_out:          Indexes of the dD
dEseqs_out:          Indexes of the dE


Outputs:
grad:                   Gradient of all trainable parameters


*/

pub fn state_space_backward_group2(
    netdata: &network_metadata_type,
    X: &arrayfire::Array<f32>,



    network_params: &arrayfire::Array<f32>,




    Z: &arrayfire::Array<f32>,
    Q: &arrayfire::Array<f32>,
    Y: &arrayfire::Array<f32>,
    loss_grad: impl Fn(&arrayfire::Array<f32>, &arrayfire::Array<f32>) -> arrayfire::Array<f32>,
    neuron_idx: &arrayfire::Array<i32>,



    idxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    valsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    cvec_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    dXsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    
    nrows_out: &nohash_hasher::IntMap<i64, u64 >,
    sparseval_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparserow_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    sparsecol_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,



    Hidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Aidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Bidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Cidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Didxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    Eidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,
    combidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >,




    dAseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dBseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dCseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dDseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,
    dEseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >,




    grad: &mut arrayfire::Array<f32>,
) {
    let neuron_size: u64 = netdata.neuron_size.clone();
    let input_size: u64 = netdata.input_size.clone();
    let output_size: u64 = netdata.output_size.clone();
    let proc_num: u64 = netdata.proc_num.clone();

    println!("---- Start the backward pass ---- \n");

    let batch_size: u64 = netdata.batch_size.clone();



    // Set output to zero
    *grad = arrayfire::constant::<f32>(0.0,network_params.dims());








    /// Get current selection of neurons
    let active_size = neuron_idx.dims()[0];
    let idxsel = arrayfire::rows(neuron_idx, (active_size-output_size) as i64, (active_size-1)   as i64);


    let Qslices: u64 = Q.dims()[2];
    let Yslices: u64 = Y.dims()[2];
    // println!("|network_f32| Value of Qslices (traj_steps): {:?}",Qslices);
    // println!("|network_f32| Value of Yslices (traj_size): {:?}",Yslices);
    /// Get Yhat
    let mut idxrs = arrayfire::Indexer::default();
    let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
    let seq2 = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
    idxrs.set_index(&idxsel, 0, None);
    idxrs.set_index(&seq1, 1, None);
    idxrs.set_index(&seq2, 2, None);
    let Yhat = arrayfire::index_gen(Q, idxrs);
    
    /// Q[neuron_size,batch_size,traj_steps,1]
    // af_print!("Value of Q: ", Q);

    /// Yhat[output_neurons,batch_size,1,1]
    // af_print!("Value of Yhat: ", Yhat);
    // NOTE: Don't drop idxsel here - it's needed later in the loop

    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let S_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,1,1]);
    
    // println!("Dimension of S: {:?}",S_dims);

    //Calculate error
    let total_error = loss_grad(&Yhat,Y);
    let mut yslicidx: i64 = (Yslices-1) as i64;
    let mut error = arrayfire::slice(&total_error, yslicidx);

    // let label = format!("error after slicing index {} along the 3rd dimension of total error", yslicidx);
    // af_print!(&label,&error);

    /// Z[neuron_size,batch_size,traj_steps]
    let Zslices: i64 = Z.dims()[2] as i64;

    /// X[input_size,batch_size,traj_steps]
    let X_slices: i64 = X.dims()[2] as i64;

    



    let mut inx = arrayfire::constant::<f32>(0.0,S_dims);

    /// AF::array[dim0,dim1,dim2,dim3] = [rows,columns,slices,batches], in a 3D cube, slices is like h
    let mut tempx =  arrayfire::slice(X, 0); // extracting the first slice of X matrix

    /// create a sequence from 0 to 3 for tempx
    let seqs = &[arrayfire::Seq::new(0.0f32, (input_size-1) as f32, 1.0f32),  arrayfire::Seq::default()];


    let mut Xtemp = arrayfire::slice(Z, 0); // extracting the first slice of Z matrix

    /// LOOK UP VALUE OF A,B,C,D,E
    let mut sA = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sB = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sC = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sD = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut sE = arrayfire::constant::<f32>(0.0,temp_dims);



    /// PARTIAL DERIVATIVES
    let mut dX = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dA = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dB = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dC = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dD = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut dE = arrayfire::constant::<f32>(0.0,temp_dims);



    /// TEMPORARY VALUE OF W
    let mut tempW = arrayfire::constant::<f32>(0.0,temp_dims);

    let mut gtemperr = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempinx = arrayfire::constant::<f32>(0.0,temp_dims);


    /// temporary derivatives of X,W,H
    let mut tempdX = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempgW = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tempgH = arrayfire::constant::<f32>(0.0,temp_dims);



    /** extract total error by first slice, Why?
     bc we get initial error grads for 1st time step
     starting point for backward prop
     **/

    /* 
    total_error[3, 5, 1, 1]:
Slice 0: [error_neuron_0_sample_0, error_neuron_1_sample_0, error_neuron_2_sample_0]
Slice 1: [error_neuron_0_sample_1, error_neuron_1_sample_1, error_neuron_2_sample_1]  
Slice 2: [error_neuron_0_sample_2, error_neuron_1_sample_2, error_neuron_2_sample_2]
Slice 3: [error_neuron_0_sample_3, error_neuron_1_sample_3, error_neuron_2_sample_3]
Slice 4: [error_neuron_0_sample_4, error_neuron_1_sample_4, error_neuron_2_sample_4]
    
    derror[3,1,1,1]:
    [error_neuron_0_sample_0, error_neuron_1_sample_0, error_neuron_2_sample_0]
*/
    let mut derror = arrayfire::slice(&total_error,  0);





    /// batchseq: select all batch samples for processing
    let batchseq = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
    /// proc_num-1 = Qslices-1 --> only time step 1 is processed
    let mut sliceseq = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);




    let mut keys = arrayfire::constant::<i32>(0,temp_dims);
    let mut vals = arrayfire::constant::<f32>(0.0,temp_dims);




    let mut UAFgroup = arrayfire::constant::<f32>(0.0,temp_dims);
    let mut tileerror = arrayfire::constant::<f32>(0.0,temp_dims);
    let tileerror_dims = arrayfire::Dim4::new(&[5,1,1,1]);



    //Main loop
    for i in (0i64..Zslices).rev() {
        println!(" Start main loop\n");
        println!("|network_f32| Timestep: {:?} \n", i);

        println!("Number of current active neurons: {:?}",active_size);
        // af_print!("Index of output neurons:", idxsel);

        // af_print!("Z value: ",Z);
        let mut idxrs = arrayfire::Indexer::default();
        sliceseq = arrayfire::Seq::new(i as f32, i as f32, 1.0);
        idxrs.set_index(&idxsel_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        idxrs.set_index(&sliceseq, 2, None);
        Xtemp = arrayfire::index_gen(Z, idxrs);

        //println!("Value of indexer object used to slice/extract Z: {:?}",idxrs);
        // af_print!("Value of temporary X: ", Xtemp);

        // Get current UAF parameters
        sA = arrayfire::lookup(&network_params, &Aidxsel_out[&i], 0);

        sB = arrayfire::lookup(&network_params, &Bidxsel_out[&i], 0);

        sC = arrayfire::lookup(&network_params, &Cidxsel_out[&i], 0);

        sD = arrayfire::lookup(&network_params, &Didxsel_out[&i], 0);

        sE = arrayfire::lookup(&network_params, &Eidxsel_out[&i], 0);







        // af_print!("Value of dX before element wise product 1: ", dX);
        // af_print!("Value of dA before UAF: ", dA);

        /// Compute derivative of UAF
        deriUAF(&Xtemp,
            &sA,
            &sB,
            &sC,
            &sD,
            &sE,
            &mut dX,
            &mut dA,
            &mut dB,
            &mut dC,
            &mut dD,
            &mut dE);
        drop(Xtemp);
        drop(sA);
        drop(sB);
        drop(sC);
        drop(sD);
        drop(sE);
        
        // af_print!("Value of dA after UAF: ", dA);
        
        // af_print!("Value of dX before element wise product 2: ", dX);
        // af_print!("Value of error: ", error);
        // af_print!("Total error grads for output neurons: ", total_error);

        // Compute dX
        dX = arrayfire::mul(&dX, &error, false);
        
        // af_print!("Value of dX after element wise product dX ⊙ error: ", dX);
        
        /// extract grad by Hidxsel_out[&i] index [3,1,1,1]
        let a = arrayfire::lookup(grad, &Hidxsel_out[&i], 0);

        // sum dX across columns
        /* 
        dX dX dX
        dX dX dX
        --> b=[3dX]
              [3dX]
        */

        let b = arrayfire::sum(&dX, 1);
        // af_print!("Current gradient from lookup operation:",a);
        // af_print!("Summed gradient across batch dimension:",b);
        
        // Update H = gradient (dX) + bias
        tempgH = arrayfire::lookup(grad, &Hidxsel_out[&i], 0) + (arrayfire::sum(&dX, 1) );
        
        // af_print!("Temporary ∂H",tempgH);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&Hidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgH);
        drop(tempgH);

        // af_print!(" Updated ∂H",grad);

        // Join all

        // Create an array with dim0 = dimdA*5 to store all partial derivatives of UAF
        UAFgroup = arrayfire::constant::<f32>(0.0,arrayfire::Dim4::new(&[dA.dims()[0]*5 , dA.dims()[1],1,1]));
        
        // add each dA,dB,dC,dD,dE with corresponding to target neurons to UAFgroup
        arrayfire::assign_seq(&mut UAFgroup, &dAseqs_out[&i], &dA);
        arrayfire::assign_seq(&mut UAFgroup, &dBseqs_out[&i], &dB);
        arrayfire::assign_seq(&mut UAFgroup, &dCseqs_out[&i], &dC);
        arrayfire::assign_seq(&mut UAFgroup, &dDseqs_out[&i], &dD);
        arrayfire::assign_seq(&mut UAFgroup, &dEseqs_out[&i], &dE);
        // UAFgrroup = 15

        // repeat error 5 times and save it in tilerror AF Array
        // error [idxsel, batch_size, traj_steps, 1]
        tileerror =  arrayfire::tile(&error, tileerror_dims);
        // af_print!("UAF group array before element wise product:",UAFgroup);
        // af_print!("Tileerror:",tileerror);
        UAFgroup = arrayfire::mul(&tileerror, &UAFgroup, false);
        drop(tileerror);
        
        // af_print!("UAF group array after element wise product UAFgroup ⊙ tileerror:",UAFgroup);
        let c = arrayfire::sum(&UAFgroup, 1);
        let d = arrayfire::lookup(grad, &combidxsel_out[&i],0);
        // af_print!("Summed UAF group gradients:",c);
        // af_print!("Current combined parameter gradient from lookup operation:",d);
        
        // Update new_grad = gradient (dX) + bias + UAFgroup
        UAFgroup = arrayfire::sum(&UAFgroup, 1) + arrayfire::lookup(grad, &combidxsel_out[&i],  0);
        
        // af_print!("Temporary ∂UAF",UAFgroup);
        
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&combidxsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &UAFgroup);
        drop(UAFgroup);

        // af_print!(" Updated ∂UAF",grad);










        /// Get dX of each row
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&dXsel_out[&i], 0, None); // extract number of rows. Eg: dXsel=[2,5,6]
        idxrs.set_index(&batchseq, 1, None); // extract number of col. Bc batchseq=5 --> select all
        tempdX = arrayfire::index_gen(&dX, idxrs);
        
        //af_print!("Temporary dX need for calculating gW:",tempdX);








/* Input reconstruction for backpropagation through time (BPTT).
   
   Since we're processing timesteps in reverse order (i = T-1, T-2, ..., 0),
   we need to reconstruct the input that was used during the forward pass
   for timestep i to compute gradients.
   
   For timestep i, the forward pass input was:
   - External input: X[i]
   - Recurrent input: Q[i-1] (from previous timestep)
   
   So inx = X[i] + Q[i-1] represents the complete input
   that was fed into the network at timestep i.
   
   Note: When i = 0, there is no previous timestep, so Q[i-1] = 0.
   */

        //Get input values
        inx = arrayfire::constant::<f32>(0.0,S_dims);
        // get input values of previous timestep (first slice of Q)
        if (i > 0)
        {
            
            inx = arrayfire::slice(Q, (i-1) );
            // println!("----- Trigger condition 1 ---- \n");
            // af_print!("Initial inx values: ",inx);
        }

        // X_slices = traj_steps
        if X_slices > 1
        {
            tempx =  arrayfire::slice(X, i);
            // println!("----- Trigger condition 2 ---- \n");
            // af_print!("Temporary Input values: ",tempx);
            // this I think is like eq 24
            arrayfire::assign_seq(&mut inx, seqs, &tempx);
            drop(tempx);
            // af_print!("Assigned Input values: ",inx);
        }
        else
        {   
            println!("----- traj_steps<1 --> Trigger condition 3 ---- \n");
            arrayfire::assign_seq(&mut inx, seqs, &X);
            //af_print!("Assigned Input values: ",inx);
        }

        // Update gW
        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&cvec_out[&i], 0, None);
        idxrs.set_index(&batchseq, 1, None);
        tempinx = arrayfire::index_gen(&inx, idxrs);
        drop(inx);
        // af_print!("Input values after extracted for calculating gW: ",tempinx);
        // af_print!("Temporary dX used for calculating gW: ",tempdX);
        tempgW = arrayfire::mul(&tempdX, &tempinx, false);
        // af_print!("gradient of W after element wise product tempdX ⊙ tempinx: ",tempgW);
        drop(tempinx);

        let e = arrayfire::sum(&tempgW, 1);
        let f = arrayfire::lookup(grad, &valsel_out[&i], 0);

        // af_print!("Summed weight gradients:",e);
        // af_print!("Current weight parameter gradient from lookup operation:",f);
        tempgW = (arrayfire::sum(&tempgW, 1) )+ arrayfire::lookup(grad, &valsel_out[&i], 0);

        // af_print!("Temporary gW: ",tempgW);

        let mut idxrs = arrayfire::Indexer::default();
        idxrs.set_index(&valsel_out[&i], 0, None);
        arrayfire::assign_gen(grad, &idxrs, &tempgW);
        drop(tempgW);

        // af_print!("Updated gW to gradient array: ",grad);

        


        // Propagate Errors
        tempW = arrayfire::lookup(network_params, &sparseval_out[&i], 0);
        // af_print!("Temporary weight initially: ",tempW);

        tempW = arrayfire::sparse::<f32>(
            nrows_out[&i],
            dX.dims()[0],
            &tempW,
            &sparserow_out[&i],
            &sparsecol_out[&i],
            arrayfire::SparseFormat::CSR
        );
        // af_print!("Temporary weight after: ",tempW);
        
        // af_print!("dX before matmul: ",dX);
        // af_print!("tempdX before matmul: ",tempdX);
        error = arrayfire::matmul(&tempW,
            &dX,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE
        );
        
        drop(tempW);
        // af_print!("Error = W . dX: ",error);



        // Add new Y error
        if (yslicidx > 0)
        {
            yslicidx = yslicidx - 1;
            derror = arrayfire::slice(&total_error,  yslicidx);
            //af_print!("derror: ",derror);

            error = arrayfire::join(0, &error, &derror);
            //af_print!("Combined error: ",error);
        }



}





}






























