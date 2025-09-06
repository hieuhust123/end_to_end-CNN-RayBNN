// We need to link `blas_src` directly, c.f. https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration
extern crate blas_src;

use numpy::{self, IntoPyArray, PyArrayDyn, ToPyArray};
use numpy::ndarray::{Zip, Array, Ix2};
use numpy::{PyReadonlyArray3, PyArray4, PyArray2, PyReadonlyArray4, PyReadonlyArray2, PyArray, PyReadonlyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python, PyObject, PyAny, Py};
use pyo3::prelude::*;
use pyo3::IntoPy;
use pyo3::types::IntoPyDict;
use crate::arrayfire::af_print;
use numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER;
use crate::arrayfire::Seq;

use arrayfire;
use raybnn;

use pythonize::{depythonize, pythonize};

use nohash_hasher;

use ndarray::Axis;

use std::collections::HashMap;

// #[derive(Clone)]
// struct TrainingResult {
//     model: raybnn::interface::automatic_f32::arch_search_type,
//     loss_history: Vec<f32>,
// }



fn sigmoid_loss(
	yhat: &arrayfire::Array<f32>,
	y: &arrayfire::Array<f32>) -> f32
{ 
	raybnn::optimal::loss_f32::weighted_sigmoid_cross_entropy(yhat, y, 5.0) 
}




fn sigmoid_loss_grad(
	yhat: &arrayfire::Array<f32>,
	y: &arrayfire::Array<f32>) -> arrayfire::Array<f32> 
{ 
	raybnn::optimal::loss_f32::weighted_sigmoid_cross_entropy_grad(yhat, y, 5.0) 
}

// fn softmax_loss(
// 	yhat: &arrayfire::Array<f32>,
// 	y: &arrayfire::Array<f32>
// ) -> f32 {
// 	raybnn::optimal::loss_f32::softmax_cross_entropy(yhat,y)
// }

// fn softmax_loss_grad(
// 	yhat: &arrayfire::Array<f32>,
// 	y: &arrayfire::Array<f32>
// ) -> arrayfire::Array<f32> {
// 	raybnn::optimal::loss_f32::softmax_cross_entropy_grad(yhat, y)
// }


const LARGE_POS_NUM_f32: f32 = 1.0e9;


#[pymodule]
fn raybnn_python<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {

	#[pyfn(m)]
    fn print_model_info<'py>(
        py: Python<'py>,
		model: Py<PyAny>
    ) {
		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		raybnn::neural::network_f32::print_netdata(&arch_search.neural_network.netdata);

		println!("WValues.dims()[0] {}",arch_search.neural_network.WColIdx.dims()[0]);
	}


	#[pyfn(m)]
    fn add_neuron_to_existing3<'py>(
        py: Python<'py>,

		new_active_size: u64,
		init_connection_num: u64,
		input_neuron_con_rad: f32,
		hidden_neuron_con_rad: f32,
		output_neuron_con_rad: f32,

		model: Py<PyAny>
    ) -> Py<PyAny>{
		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();


		//Add 30 neurons to existing neural network
		//Raytrace radius of 40 neuron radius
		let add_neuron_options: raybnn::physics::update_f32::add_neuron_option_type = raybnn::physics::update_f32::add_neuron_option_type {
			new_active_size: new_active_size,
			init_connection_num: init_connection_num,
			input_neuron_con_rad: input_neuron_con_rad,
			hidden_neuron_con_rad: hidden_neuron_con_rad,
			output_neuron_con_rad: output_neuron_con_rad,
		};

		//Add 30 neurons to existing neural network
		raybnn::physics::update_f32::add_neuron_to_existing3(
			&add_neuron_options,
			&mut arch_search,
		);


		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}




	#[pyfn(m)]
    fn select_forward_sphere<'py>(
        py: Python<'py>,


		model: Py<PyAny>
    ) -> Py<PyAny>{
		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);






		let WValuesdims0 =  (arch_search).neural_network.WColIdx.dims()[0];

		let network_paramsdims0 =  (arch_search).neural_network.network_params.dims()[0];
	
		let Hdims0 =  (network_paramsdims0 -  WValuesdims0)/6; 
	
	
	
		let Wstart = 0;
		let Wend = (WValuesdims0  as i64) - 1;
	
		let Hstart = Wend + 1; 
		let Hend = Hstart + (Hdims0 as i64) - 1;
	
		let Astart = Hend + 1; 
		let Aend = Astart + (Hdims0 as i64) - 1;
	
		let Bstart = Aend + 1; 
		let Bend = Bstart + (Hdims0 as i64) - 1;
	
		let Cstart = Bend + 1; 
		let Cend = Cstart + (Hdims0 as i64) - 1;
	
		let Dstart = Cend + 1; 
		let Dend = Dstart + (Hdims0 as i64) - 1;
	
		let Estart = Dend + 1; 
		let Eend = Estart + (Hdims0 as i64) - 1;
	
	
		let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
		let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
		let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
		let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
		let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
		let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
		let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];
	
		
	
		let mut WValues = arrayfire::index(&((arch_search).neural_network.network_params), &Wseqs);
		let H = arrayfire::index(&((arch_search).neural_network.network_params), &Hseqs);
		let A = arrayfire::index(&((arch_search).neural_network.network_params), &Aseqs);
		let B = arrayfire::index(&((arch_search).neural_network.network_params), &Bseqs);
		let C = arrayfire::index(&((arch_search).neural_network.network_params), &Cseqs);
		let D = arrayfire::index(&((arch_search).neural_network.network_params), &Dseqs);
		let E = arrayfire::index(&((arch_search).neural_network.network_params), &Eseqs);
	
	






		raybnn::graph::adjacency_f32::select_forward_sphere(
			&arch_search.neural_network.netdata, 
			&mut WValues, 
			&mut WRowIdxCOO, 
			&mut arch_search.neural_network.WColIdx, 
			&arch_search.neural_network.neuron_pos, 
			&arch_search.neural_network.neuron_idx
		);

		arch_search.neural_network.WRowIdxCSR = raybnn::graph::large_sparse_i32::COO_to_CSR(&WRowIdxCOO,arch_search.neural_network.netdata.neuron_size);






		let total_param_size = WValues.dims()[0]  +  H.dims()[0]  +  A.dims()[0]    +  B.dims()[0]    +  C.dims()[0]    +  D.dims()[0]   +  E.dims()[0]  ;
		let network_params_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
	
		let Wstart = 0;
		let Wend = (WValues.dims()[0]  as i64) - 1;
	
		let Hstart = Wend + 1; 
		let Hend = Hstart + (H.dims()[0] as i64) - 1;
	
		let Astart = Hend + 1; 
		let Aend = Astart + (A.dims()[0] as i64) - 1;
	
		let Bstart = Aend + 1; 
		let Bend = Bstart + (B.dims()[0] as i64) - 1;
	
		let Cstart = Bend + 1; 
		let Cend = Cstart + (C.dims()[0] as i64) - 1;
	
		let Dstart = Cend + 1; 
		let Dend = Dstart + (D.dims()[0] as i64) - 1;
	
		let Estart = Dend + 1; 
		let Eend = Estart + (E.dims()[0] as i64) - 1;
	
	
		let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
		let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
		let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
		let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
		let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
		let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
		let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];
	
	
		(arch_search).neural_network.network_params = arrayfire::constant::<f32>(0.0,network_params_dims);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Wseqs, &WValues);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Hseqs, &H);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Aseqs, &A);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Bseqs, &B);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Cseqs, &C);
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Dseqs, &D);	
		arrayfire::assign_seq(&mut ((arch_search).neural_network.network_params), &Eseqs, &E);	
	
	
	
		(arch_search).neural_network.netdata.active_size = (arch_search).neural_network.neuron_idx.dims()[0];
	




		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}



	#[pyfn(m)]
    fn create_start_archtecture<'py>(
        py: Python<'py>,
		input_size: u64,
		max_input_size: u64,

		output_size: u64,
		max_output_size: u64,

		active_size: u64,
		max_neuron_size: u64,

		batch_size: u64,
		traj_size: u64,

		proc_num: u64,

		directory_path:  String
    ) -> Py<PyAny> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let dir_path = directory_path.clone();

		let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture2(

			input_size,
			max_input_size,

			output_size,
			max_output_size,

			active_size,
			max_neuron_size,

			batch_size,
			traj_size,

			proc_num,

			&dir_path
		);

		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}




	#[pyfn(m)]
    fn train_network<'py>(
        py: Python<'py>,

		train_x: PyReadonlyArray4<'py, f32>,
        train_y: PyReadonlyArray4<'py, f32>,

		crossval_x: PyReadonlyArray4<'py, f32>,
        crossval_y: PyReadonlyArray4<'py, f32>,

		stop_strategy_input: String,
		lr_strategy_input: String,
		lr_strategy2_input: String,

		loss_function: String,

		max_epoch: u64,
		stop_epoch: u64,
		stop_train_loss: f32,

		max_alpha: f32,

		exit_counter_threshold: u64,
		shuffle_counter_threshold: u64,

		model: Py<PyAny>
    ) -> Py<PyAny> {

		let mut stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::NONE;

		if stop_strategy_input == "NONE"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::NONE;
		}
		else if stop_strategy_input == "STOP_AT_EPOCH"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::STOP_AT_EPOCH;
		}
		else if stop_strategy_input == "STOP_AT_TRAIN_LOSS"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::STOP_AT_TRAIN_LOSS;
		}
		else if stop_strategy_input == "CROSSVAL_STOPPING"
		{
			stop_stategy = raybnn::interface::autotrain_f32::stop_strategy_type::CROSSVAL_STOPPING;
		}




		let mut lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::NONE;

		if lr_strategy_input == "NONE"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::NONE;
		}
		else if lr_strategy_input == "COSINE_ANNEALING"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::COSINE_ANNEALING;
		}
		else if lr_strategy_input == "SHUFFLE_CONNECTIONS"
		{
			lr_strategy = raybnn::interface::autotrain_f32::lr_strategy_type::SHUFFLE_CONNECTIONS;
		}


		let mut lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::BTLS_ALPHA;

		if lr_strategy2_input == "BTLS_ALPHA"
		{
			lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::BTLS_ALPHA;
		}
		else if lr_strategy2_input == "MAX_ALPHA"
		{
			lr_strategy2 = raybnn::interface::autotrain_f32::lr_strategy2_type::MAX_ALPHA;
		}




		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		//Train Options
		let train_stop_options = raybnn::interface::autotrain_f32::train_network_options_type {
			stop_strategy: stop_stategy,
			lr_strategy: lr_strategy,
			lr_strategy2: lr_strategy2,

			max_epoch: max_epoch,
			stop_epoch: stop_epoch,
			stop_train_loss: stop_train_loss,

			exit_counter_threshold: exit_counter_threshold,
			shuffle_counter_threshold: shuffle_counter_threshold,
		};


		let mut alpha_max_vec = vec![max_alpha; 1000];
		let mut loss_vec = Vec::new();
		let mut crossval_vec = Vec::new();
		let mut loss_status = raybnn::interface::autotrain_f32::loss_status_type::LOSS_OVERFLOW;

		println!("Start training");

		arrayfire::device_gc();





		let train_x_dims = train_x.shape().to_vec();
		println!("[TRAIN_NETWORK_lib.rs] Input train X dims: {:?}", train_x_dims);
		let train_y_dims = train_y.shape().to_vec();

		let crossval_x_dims = crossval_x.shape().to_vec();
		let crossval_y_dims = crossval_y.shape().to_vec();

		let mut traindata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut traindata_Y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();

		let mut validationdata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut validationdata_Y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();


		let train_x = train_x.to_owned_array();
		// Print only shape and first few elements
		let sample_data: Vec<f32> = train_x.clone().into_iter().take(50).collect();
		println!("[TRAIN_NETWORK_lib.rs] Input train X first 50 elements in Rust ndarray: {:?}", sample_data);
		
		let train_y = train_y.to_owned_array() ;

		let crossval_x = crossval_x.to_owned_array() ;
		let crossval_y = crossval_y.to_owned_array() ;


		let Xslices = train_x_dims[2];


		for traj in 0..train_x_dims[3]
		{


			let train_x_dims = train_x.shape().to_vec();
			let train_y_dims = train_y.shape().to_vec();

			let mut X = Vec::new();
			let mut Y = Vec::new();
			if Xslices > 1
			{
				X = train_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_x_dims[0],train_x_dims[2],train_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				Y = train_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_y_dims[0],train_y_dims[2],train_y_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();	
			}
			else
			{
				X = train_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_x_dims[1],train_x_dims[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				Y = train_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([train_y_dims[1],train_y_dims[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();	
			}




			traindata_X.insert(traj as u64, X);
			traindata_Y.insert(traj as u64, Y);
		}
		// Print HashMap summary instead of full content
		println!("[TRAIN_NETWORK_lib.rs] Input train X HashMap size: {}", traindata_X.len());
		if let Some(first_traj) = traindata_X.get(&0) {
			let sample_size = std::cmp::min(10, first_traj.len());
			println!("[TRAIN_NETWORK_lib.rs] First trajectory (traj=0) first {} elements: {:?}", sample_size, &first_traj[0..sample_size]);
		}
		if let Some(last_traj) = traindata_X.get(&(traindata_X.len() as u64 - 1)) {
			let sample_size = std::cmp::min(10, last_traj.len());
			println!("[TRAIN_NETWORK_lib.rs] Last trajectory (traj={}) first {} elements: {:?}", traindata_X.len()-1, sample_size, &last_traj[0..sample_size]);
		}
		for traj in 0..crossval_x_dims[3]
		{
			let crossval_x_dims = crossval_x.shape().to_vec();
			let crossval_y_dims = crossval_y.shape().to_vec();

			let mut X = Vec::new();
			let mut Y = Vec::new();
			if Xslices > 1
			{
				X = crossval_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_x_dims[0],crossval_x_dims[2],crossval_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				Y = crossval_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_y_dims[0],crossval_y_dims[2],crossval_y_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}
			else 
			{
				X = crossval_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_x_dims[1],crossval_x_dims[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				Y = crossval_y.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([crossval_y_dims[1],crossval_y_dims[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}


			validationdata_X.insert(traj as u64, X);
			validationdata_Y.insert(traj as u64, Y);
		}

		// // ADD THESE PRINT STATEMENTS TO SHOW FINAL PROCESSED DATA:
		// println!("traindata_X HashMap created with {} trajectories", traindata_X.len());
		// println!("First trajectory data length: {:?}", traindata_X.get(&0).map(|v| v.len()));
		
		// // Show sample of first trajectory data (first 10 elements)
		// if let Some(first_traj) = traindata_X.get(&0) {
		// 	let sample_size = std::cmp::min(10, first_traj.len());
		// 	println!("First trajectory sample data: {:?}", &first_traj[0..sample_size]);
		// }

		if loss_function == "MSE"
		{
			//Train network, stop at lowest crossval
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::MSE,
				raybnn::optimal::loss_f32::MSE_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}
		else if loss_function == "softmax_cross_entropy"
		{
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::softmax_cross_entropy,
				raybnn::optimal::loss_f32::softmax_cross_entropy_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}
		else if loss_function == "sigmoid_cross_entropy"
		{
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				raybnn::optimal::loss_f32::sigmoid_cross_entropy,
				raybnn::optimal::loss_f32::sigmoid_cross_entropy_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}
		else if loss_function == "sigmoid_cross_entropy_5"
		{
			raybnn::interface::autotrain_f32::train_network(
				&traindata_X,
				&traindata_Y,

				&validationdata_X,
				&validationdata_Y,

				sigmoid_loss,
				sigmoid_loss_grad,

				train_stop_options,

				&mut alpha_max_vec,
				&mut loss_vec,
				&mut crossval_vec,
				&mut arch_search,
				&mut loss_status
			);
		}

		let obj = pythonize(py, &arch_search).unwrap();

		obj
	}







	#[pyfn(m)]
    fn test_network<'py>(
        py: Python<'py>,

		test_x: PyReadonlyArray4<'py, f32>,


		model: Py<PyAny>
    ) -> &'py PyArray4<f32>   {



		arrayfire::set_backend(arrayfire::Backend::CUDA);
		arrayfire::device_gc();

		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();


		let test_x_dims = test_x.shape().to_vec();


		let mut validationdata_X: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();


		let test_x = test_x.to_owned_array() ;


		let Xslices = test_x_dims[2];

		
		for traj in 0..test_x_dims[3]
		{
			let test_x_dims = test_x.shape().to_vec();

			let mut X = Vec::new();

			if Xslices > 1
			{
				X = test_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([test_x_dims[0],test_x_dims[2],test_x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}
			else 
			{
				X = test_x.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([test_x_dims[1],test_x_dims[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}


			validationdata_X.insert(traj as u64, X);
			
		}

		
		let mut Yhat_out = nohash_hasher::IntMap::default();
		

		//Train network, stop at lowest crossval
		raybnn::interface::autotest_f32::test_network(
			&mut validationdata_X,

			&mut arch_search, 

			&mut Yhat_out, 
		);

		/* 
		arrayfire::set_seed(0);
		let Yhat_dims = arrayfire::Dim4::new(&[2, 3, 5, 7]);
		let Yhat_arr = arrayfire::randu::<f32>(Yhat_dims);

		arrayfire::print_gen("Yhat_arr".to_string(), &Yhat_arr,Some(6));

		let mut Yhat_vec = vec!(f32::default();Yhat_arr.elements());
		Yhat_arr.host(&mut Yhat_vec);
		*/

		let arr = unsafe {

			let dim0 = arch_search.neural_network.netdata.output_size as usize;
			let dim1 = test_x_dims[1] as usize;
			let dim2 = test_x_dims[2] as usize;
			let dim3 = test_x_dims[3] as usize;

			let arr = PyArray4::<f32>::new(py, [dim0, dim1, dim2, dim3], true);


			for l in 0..dim3
			{
				let idx = l as u64;
				let Yhat_vec = Yhat_out[&idx].clone();


				for i in 0..dim0 {
					for j in 0..dim1 {
						for k in 0..dim2 {
							arr.uget_raw([i, j, k, l]).write(Yhat_vec[i + (dim0*j) + (dim0*dim1*k) ]);
						}
					}
				}

			}

			

			arr
		};



		arr
	}






	// same as rows_dot
	#[pyfn(m)]
    fn magic2<'py>(
        py: Python<'py>,
        x: PyReadonlyArray3<'py, f32>,
    ) -> Py<PyAny> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let x_dims = x.shape().to_vec();
        let x = x.reshape_with_order([x_dims[0],x_dims[2],x_dims[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[0] as u64, x_dims[1] as u64, x_dims[2] as u64, 1]));
		arrayfire::print_gen("a".to_string(), &a, Some(6));

		let mut b = arrayfire::row(&a,1);
		b = arrayfire::col(&b,0);
		b = arrayfire::slice(&b,3);
		arrayfire::print_gen("b".to_string(), &b, Some(6));

		let obj = pythonize(py, &a).unwrap();

		obj
	}

	// this is a custom Python binding fn, used x and y as input parameters
    #[pyfn(m)]
    fn rows_dot<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {

		arrayfire::set_backend(arrayfire::Backend::CUDA);

		let x_dims = x.shape().to_vec();
        let x = x.to_vec().unwrap();


		let mut a = arrayfire::Array::new(&x, arrayfire::Dim4::new(&[x_dims[1] as u64, x_dims[0] as u64, 1, 1]));
		a = arrayfire::transpose(&a, false);
		arrayfire::print_gen("a".to_string(), &a, Some(6));



		let mut output_vec: Vec<Vec<f64> >  = Vec::new() ;
		for i in 0..a.dims()[0]
		{
			let row = arrayfire::row(&a,i as i64);
			let mut tempvec = vec!(f64::default();row.elements());
			row.host(&mut tempvec);
			output_vec.push(tempvec);
		}


		let output = PyArray2::from_vec2(py, &output_vec).unwrap();
		output
    }


	#[pyfn(m)]
	fn state_space_forward_batch<'py>(
		py: Python<'py>,
		x_train_py: PyReadonlyArray4<'py, f32>,
		// x_input_py: &'py PyArray4<f32>,
		y_train_py: PyReadonlyArray4<'py, f32>,
		traj_size: u64,
		max_epoch: u64,
		model: Py<PyAny>,
	) -> Py<PyAny> {
		arrayfire::set_backend(arrayfire::Backend::CUDA);
		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();

		// declare default dims
		let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);	
		
		// init model params
		let neuron_size: u64 = arch_search.neural_network.netdata.neuron_size.clone();
		let batch_size: u64  = arch_search.neural_network.netdata.batch_size.clone();
		let proc_num: u64 = arch_search.neural_network.netdata.proc_num.clone();
		let input_size: u64 = arch_search.neural_network.netdata.input_size.clone();
		let output_size: u64 = arch_search.neural_network.netdata.output_size.clone();
		let mut active_size: u64 = arch_search.neural_network.netdata.active_size.clone();
		println!("trajectory size: {:?}", traj_size);

		let traj_steps = traj_size+proc_num-1;
		let Z_dims = arrayfire::Dim4::new(&[neuron_size,batch_size,traj_steps,1]);


		let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);
		let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);

		let mut WRowIdxCSR = &arch_search.neural_network.WRowIdxCSR;
		//let mut WColIdx = &arch_search.neural_network.WColIdx;
	
		let WValuesdims0 = (arch_search).neural_network.WColIdx.dims()[0];

		let network_params = (arch_search).neural_network.network_params.dims()[0];
		active_size = (arch_search).neural_network.neuron_idx.dims()[0];
		let idxsel = arrayfire::rows(&((arch_search).neural_network.neuron_idx), (active_size-output_size)  as i64, (active_size-1)  as i64);

		let mut loss_val = LARGE_POS_NUM_f32;
		let mut loss_history: Vec<f32> = Vec::new();

		let Hdims0 = (network_params - WValuesdims0)/6;

		let Wstart = 0;
		let Wend = (WValuesdims0  as i64) - 1;
	
		let Hstart = Wend + 1; 
		let Hend = Hstart + (Hdims0 as i64) - 1;
	
		let Astart = Hend + 1; 
		let Aend = Astart + (Hdims0 as i64) - 1;
	
		let Bstart = Aend + 1; 
		let Bend = Bstart + (Hdims0 as i64) - 1;
	
		let Cstart = Bend + 1; 
		let Cend = Cstart + (Hdims0 as i64) - 1;
	
		let Dstart = Cend + 1; 
		let Dend = Dstart + (Hdims0 as i64) - 1;
	
		let Estart = Dend + 1; 
		let Eend = Estart + (Hdims0 as i64) - 1;

		let Wseqs = [arrayfire::Seq::new(Wstart as i32, Wend as i32, 1i32)];
		let Hseqs = [arrayfire::Seq::new(Hstart as i32, Hend as i32, 1i32)];
		let Aseqs = [arrayfire::Seq::new(Astart as i32, Aend as i32, 1i32)];
		let Bseqs = [arrayfire::Seq::new(Bstart as i32, Bend as i32, 1i32)];
		let Cseqs = [arrayfire::Seq::new(Cstart as i32, Cend as i32, 1i32)];
		let Dseqs = [arrayfire::Seq::new(Dstart as i32, Dend as i32, 1i32)];
		let Eseqs = [arrayfire::Seq::new(Estart as i32, Eend as i32, 1i32)];
		
		//println!("Input X: {:?}", X);

		// Convert Input training data into Rust HashMap

		let x_train_shape = x_train_py.shape().to_vec();
		let y_train_shape = y_train_py.shape().to_vec();
		println!("[Forward pass function PyO3] Input train X dims: {:?}", x_train_shape);
		
		let x_slices = x_train_shape[2];
		let mut traindata_x: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		let mut traindata_y: nohash_hasher::IntMap<u64, Vec<f32> > = nohash_hasher::IntMap::default();
		// Rust ndarray
		let x_input_ndarray = x_train_py.to_owned_array();
		let y_input_ndarray = y_train_py.to_owned_array();
		// Print only shape and first few elements
		let sample_data: Vec<f32> = x_input_ndarray.clone().into_iter().take(50).collect();
		println!("[Forward pass function PyO3] Input train X first 50 elements in Rust ndarray: {:?}", sample_data);

		let mut train_x_af =  arrayfire::constant::<f32>(0.0,temp_dims);
		let train_x_dims = arrayfire::Dim4::new(&[input_size, batch_size, traj_size, 1]);
		let train_y_dims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
		let mut train_y_af =  arrayfire::constant::<f32>(0.0,temp_dims);

		for traj in 0..x_train_shape[3]
		{
			let mut train_x_rust_vec = Vec::new();
			let mut train_y_rust_vec = Vec::new();
			if x_slices > 1
			{
				train_x_rust_vec = x_input_ndarray.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([x_train_shape[0], x_train_shape[2],x_train_shape[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				train_y_rust_vec = y_input_ndarray.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([y_train_shape[0], y_train_shape[2],y_train_shape[1]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}
			else
			{
				train_x_rust_vec = x_input_ndarray.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([x_train_shape[1], x_train_shape[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
				train_y_rust_vec = y_input_ndarray.index_axis(Axis(3), traj).to_owned().into_pyarray(py).reshape_with_order([y_train_shape[1], y_train_shape[0]], numpy::npyffi::types::NPY_ORDER::NPY_FORTRANORDER).unwrap().to_vec().unwrap();
			}

			traindata_x.insert(traj as u64, train_x_rust_vec);
			traindata_y.insert(traj as u64, train_y_rust_vec);
		}

		// Print HashMap summary instead of full content
		println!("[Forward pass function PyO3] Input train X HashMap size: {}", traindata_x.len());
		if let Some(first_traj) = traindata_x.get(&0) {
			let sample_size = std::cmp::min(10, first_traj.len());
			println!("[Forward pass function PyO3] First trajectory (traj=0) first {} elements: {:?}", sample_size, &first_traj[0..sample_size]);
		}
		if let Some(last_traj) = traindata_x.get(&(traindata_x.len() as u64 - 1)) {
			let sample_size = std::cmp::min(10, last_traj.len());
			println!("[Forward pass function PyO3] Last trajectory (traj={}) first {} elements: {:?}", traindata_x.len()-1, sample_size, &last_traj[0..sample_size]);
		}
		// now we have traindata_X same as train_network() func

		
		let mut batch_idx = 0;
		let epoch_num = traindata_x.len() as u64;
		train_x_af = arrayfire::Array::new(&traindata_x[&batch_idx], train_x_dims);
		train_y_af = arrayfire::Array::new(&traindata_y[&batch_idx], train_y_dims);
		println!("[Forward pass function PyO3] Input arrayfire training shape: {:?}", train_x_af);
		// Print only first few elements of ArrayFire array
		let sample_elements = arrayfire::index(&train_x_af, &[Seq::new(0.0, 9.0, 1.0), Seq::default(), Seq::default(), Seq::default()]);
		// af_print!("[Forward pass function PyO3] Input arrayfire training first 10 elements: ", sample_elements);


		for i in 0..max_epoch
		{	
			batch_idx = i % epoch_num;
			train_x_af = arrayfire::Array::new(&traindata_x[&batch_idx], train_x_dims);
			train_y_af = arrayfire::Array::new(&traindata_y[&batch_idx], train_y_dims);
		// raybnn::neural::network_f32::state_space_forward_batch(
		// &arch_search.neural_network.netdata,
		// &train_x_af,
		
		// &WRowIdxCSR,
		// &mut arch_search.neural_network.WColIdx,

		// &Wseqs,
		// &Hseqs,
		// &Aseqs,
		// &Bseqs,
		// &Cseqs,
		// &Dseqs,
		// &Eseqs,

		// &arch_search.neural_network.network_params,

		// &mut Z,
		// &mut Q,
		// );

		raybnn::interface::autotrain_f32::loss_wrapper(
			&arch_search.neural_network.netdata,
			&train_x_af,
			
			&WRowIdxCSR,
			&mut arch_search.neural_network.WColIdx,
	
			&Wseqs,
			&Hseqs,
			&Aseqs,
			&Bseqs,
			&Cseqs,
			&Dseqs,
			&Eseqs,
			&arch_search.neural_network.network_params,
			&idxsel,
			&train_y_af,
			raybnn::optimal::loss_f32::softmax_cross_entropy,
			&mut Z,
			&mut Q,
			&mut loss_val
		);

		println!("Epoch {}: Loss = {:?} \n", i, loss_val);
		// loss_history.push(loss_val);
		
	}
		
		// let result = TrainingResult {
		// 	model: arch_search,
		// 	loss_history: loss_history,
		// };
		
		let obj = pythonize(py, &arch_search).unwrap();

		obj
	
	}





    // #[pyfn(m)]
    // fn state_space_backward_group2<'py>(
    //     py: Python<'py>,
    
		
    // X: &'py PyArray4<f32>,
	// Y: &'py PyArray4<f32>,

	// model: Py<PyAny>,

    // ) -> &'py PyArray4<f32> {
	// 	// set the function to run on CUDA
	// 	arrayfire::set_backend(arrayfire::Backend::CUDA);
	// 	// Initialize Rust struct
	// 	let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(model.as_ref(py)).unwrap();
		
	// 	let x_slice = X.as_slice().unwrap();
	// 	let shape = X.shape();
	// 	let af_x = arrayfire::Array::new(x_slice,arrayfire::Dim4::new(&[
			// shape[0] as u64,
			// shape[1] as u64,
			// shape[2] as u64,
			// shape[3] as u64,]),);

	// 	let y_slice = Y.as_slice().unwrap();
	// 	let shape = Y.shape();
	// 	let af_y = arrayfire::Array::new(y_slice,arrayfire::Dim4::new(&[
	// 		shape[0] as u64,
	// 		shape[1] as u64,
	// 		shape[2] as u64,
	// 		shape[3] as u64,
	// 	]),);


	// 	let  Q : &arrayfire::Array<f32>;
	// 	let  Z : &arrayfire::Array<f32>;
	// 	// let  Y = &arrayfire::Array<f32>,
	// 	//let  loss_grad = raybnn::loss_f32::softmax_cross_entropy_grad,
	// 	let  neuron_idx : &arrayfire::Array<f32>;

	// 	let	idxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	valsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
		
	// 	let	cvec_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	dXsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
		
	// 	let	nrows_out: &nohash_hasher::IntMap<i64, u64 >;
	// 	let	sparseval_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	sparserow_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	sparsecol_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	
	
	
	// 	let	Hidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	Aidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	Bidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	Cidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	Didxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	Eidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	// 	let	combidxsel_out: &nohash_hasher::IntMap<i64, arrayfire::Array<i32> >;
	
	
	
	
	// 	let	dAseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >;
	// 	let	dBseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >;
	// 	let	dCseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >;
	// 	let	dDseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >;
	// 	let	dEseqs_out: &nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] >;
	
	
	
	
	// 	let	grad: &mut arrayfire::Array<f32>;



	// raybnn::neural::network_f32::state_space_backward_group2(
	// 	&arch_search.neural_network.netdata,
	// 	&af_x,

	// 	&arch_search.neural_network.network_params,

	// 	&Z,
	// 	&Q,
	// 	&af_y,

	// 	raybnn::optimal::loss_f32::softmax_cross_entropy_grad,
	// 	&arch_search.neural_network.neuron_idx,
			
		
	// 			&idxsel_out,
	// 			&valsel_out,

	// 			&cvec_out,
	// 			&dXsel_out,

	// 			&nrows_out,
	// 			&sparseval_out,
	// 			&sparserow_out,
	// 			&sparsecol_out,

			
			
	// 			&Hidxsel_out,
	// 			&Aidxsel_out,
	// 			&Bidxsel_out,
	// 			&Cidxsel_out,
	// 			&Didxsel_out,
	// 			&Eidxsel_out,
	// 			&combidxsel_out,



	// 			&dAseqs_out,
	// 			&dBseqs_out,
	// 			&dCseqs_out,
	// 			&dDseqs_out,
	// 			&dEseqs_out,
			

				
	// 			&mut grad,
	// );
	// let grad_shape = grad.dims();
	// let mut grad_vec = vec![dims[0] as usize,
	// dims[1] as usize,
	// dims[2] as usize,
	// dims[3] as usize];
	// let grad_vec: Vec<_> = shape.into_iter().filter(|&x| x > 1).collect();

	// let mut vec = vec![0.0; grad.elements()];
	// af::copy_array(&grad).host(&mut vec);


	
	// 	// Create numpy array from grad_vec
	// 	numpy::PyArray::from_shape_vec(py, shape, grad_vec).unwrap();
	
	// 	grads
	// //let grads = PyArray4::

	// }

	#[pyfn(m)]
	fn state_space_forward_pytorch<'py>(
		py: Python<'py>,
		features_tensor: PyReadonlyArray2<'py, f32>,  // [batch_size, feature_dim]
		arch_search: Py<PyAny>,
		traj_size: u64,
		max_epoch: u64,
	) -> Py<PyAny> {
		arrayfire::set_backend(arrayfire::Backend::CUDA);
		let mut arch_search: raybnn::interface::automatic_f32::arch_search_type = depythonize(arch_search.as_ref(py)).unwrap();

		// Get tensor dimensions
		let features_shape = features_tensor.shape();
		let batch_size = features_shape[0] as u64;
		let feature_dim = features_shape[1] as u64;
		
		println!("[PyTorch RayBNN] Input features shape: [{}, {}]", batch_size, feature_dim);

		// Convert to the format expected by RayBNN
		let features_array = features_tensor.to_owned_array();
		
		// Create training data in RayBNN format: [input_size, batch_size, traj_size, num_trajectories]
		let train_x_dims = arrayfire::Dim4::new(&[feature_dim, batch_size, traj_size, 1]);
		let train_y_dims = arrayfire::Dim4::new(&[10, batch_size, traj_size, 1]);  // Assuming 10 classes
		
		// Convert numpy array to ArrayFire
		let mut train_x_af = arrayfire::constant::<f32>(0.0, train_x_dims);
		let mut train_y_af = arrayfire::constant::<f32>(0.0, train_y_dims);
		
		// Fill the training data
		for i in 0..batch_size {
			for j in 0..feature_dim {
				let idx = (j * batch_size + i) as i64;
				train_x_af = arrayfire::assign_seq(&mut train_x_af, 
					&[arrayfire::Seq::new(j as i32, j as i32, 1), 
					  arrayfire::Seq::new(i as i32, i as i32, 1), 
					  arrayfire::Seq::new(0, 0, 1), 
					  arrayfire::Seq::new(0, 0, 1)], 
					&arrayfire::constant::<f32>(features_array[[i as usize, j as usize]], arrayfire::Dim4::new(&[1,1,1,1])));
			}
		}

		// Call the existing forward pass logic
		let mut Z = arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size, batch_size, traj_size + arch_search.neural_network.netdata.proc_num - 1, 1]));
		let mut Q = arrayfire::constant::<f32>(0.0, arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size, batch_size, traj_size + arch_search.neural_network.netdata.proc_num - 1, 1]));
		
		let mut loss_val = LARGE_POS_NUM_f32;
		
		// Perform forward pass
		raybnn::neural::network_f32::state_space_forward_batch(
			&arch_search.neural_network.netdata,
			&train_x_af,
			&arch_search.neural_network.WRowIdxCSR,
			&mut arch_search.neural_network.WColIdx,
			&[arrayfire::Seq::default()],  // Wseqs
			&[arrayfire::Seq::default()],  // Hseqs
			&[arrayfire::Seq::default()],  // Aseqs
			&[arrayfire::Seq::default()],  // Bseqs
			&[arrayfire::Seq::default()],  // Cseqs
			&[arrayfire::Seq::default()],  // Dseqs
			&[arrayfire::Seq::default()],  // Eseqs
			&arch_search.neural_network.network_params,
			&mut Z,
			&mut Q
		);

		// Extract output (you'll need to modify this based on your actual output format)
		let output_dims = arrayfire::Dim4::new(&[10, batch_size, 1, 1]);  // Assuming 10 classes
		let output = arrayfire::constant::<f32>(0.0, output_dims);  // Placeholder
		
		// Convert back to PyTorch tensor format
		let output_array = PyArray2::<f32>::new(py, [batch_size as usize, 10], true);
		
		// Fill output array (placeholder - you'll need to extract actual output from Q)
		for i in 0..batch_size as usize {
			for j in 0..10 {
				output_array.uget_raw([i, j]).write(0.0);  // Placeholder value
			}
		}

		output_array.into()
	}

	Ok(())
}
