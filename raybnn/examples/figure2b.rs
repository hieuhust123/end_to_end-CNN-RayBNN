/*
Plot Figure 2b Measuring runtime of various collision detection algorithms

The code runs serial, mini-batch, and batch versions of cell collision detection.
It compares the runtimes of those algorithms


Generates these files

./collision_run_time.csv           Contains the time it takes to run collision detection by looking at minibatches of cells
./collision_run_time_serial.csv    Contains the time it takes to run collision detection by looking at cells  one by one
./collision_run_time_batch.csv     Contains the time it takes to run collision detection by looking at all cells at once


*/










extern crate arrayfire;
extern crate raybnn;




use std::time::{Duration, Instant};

// Use CUDA GPU and GPU device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


#[allow(unused_must_use)]
fn main() {

	// Use CUDA GPU and GPU device 0
	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);


	//Batch Collision Detection Version


	//List of Neuron Sizes to sweep through
	let active_size_list = vec![10,10,20,40,80,160,320,640,1280,2000,4000,8000];


	//Create Initial Neural Network

	let neuron_size: u64 = 51000;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 3;
	let active_size: u64 = 500000;
	let space_dims: u64 = 3;
	let sim_steps: u64 = 1;
	let mut batch_size: u64 = 105;
	let neuron_rad = 1.0; //Cell Radius
	let sphere_rad = 600.0; //Neural Network Radius

	let mut netdata: raybnn::neural::network_f32::network_metadata_type = raybnn::neural::network_f32::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: sim_steps,
		batch_size: batch_size,
		del_unused_neuron: true,

		time_step: 0.3,
		nratio: 0.5,
		neuron_std: 0.3,
		sphere_rad: sphere_rad,
		neuron_rad: neuron_rad,
		con_rad: 0.6,
		init_prob: 0.5,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 0.01
	};

	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	arrayfire::sync(DEVICE);

	let mut final_neuron_num = Vec::new();
	let mut final_glia_num = Vec::new();

	let mut collision_run_time = Vec::new();


	let mut initial_cell_num = active_size_list.clone();
	initial_cell_num.iter_mut().for_each(|b| *b = (*b * 2));

	println!("Initial Number of Cells: {:?}", initial_cell_num);

	


	//Loop Through Different Neural Network Sizes
	arrayfire::sync(DEVICE);
	for ii in 0..10
	{
		for new_active_size in active_size_list.clone()
		{
			netdata.active_size = new_active_size.clone();

			//println!("new_active_size {}",new_active_size);
	
			let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
			let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);
		
			arrayfire::sync(DEVICE);
			let start = Instant::now();
	
			//Create Neural Network Sphere
			raybnn::physics::initial_f32::sphere_cell_collision_batch(
				&netdata,
				&mut glia_pos,
				&mut neuron_pos
			);
		
			//Record the number of neural network cells

			final_neuron_num.push(neuron_pos.dims()[0]);
			final_glia_num.push(glia_pos.dims()[0]);
	
			let duration = start.elapsed();
	
			collision_run_time.push(duration.as_secs_f64());
	
		}
		std::thread::sleep(std::time::Duration::from_secs(10));
	}

	//Print the final number of neural network cells after deleting collided cells
    
	println!("final_neuron_num {:?}",final_neuron_num);
	println!("final_glia_num {:?}",final_glia_num);
	println!("collision_run_time {:?}",collision_run_time);


	let mut collision_run_time_f64 = Vec::new();
	for element in collision_run_time.clone() { 
		collision_run_time_f64.push(element as f64);
	}
	//Save Collision Runtime
	raybnn::export::dataloader_f64::write_vec_cpu_to_csv(
		"./collision_run_time_batch.csv",
		&collision_run_time_f64
	);














	//Serial Collision Detection Version

	//List of Neuron Sizes to sweep through
	let active_size_list = vec![10,10,20,40,80,160,320,640,1280,2000,4000,8000,16000,32000,64000,128000,256000,512000];


	//Create Initial Neural Network

	let neuron_size: u64 = 51000;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 3;
	let active_size: u64 = 500000;
	let space_dims: u64 = 3;
	let sim_steps: u64 = 1;
	let mut batch_size: u64 = 105;
	let neuron_rad = 1.0; //Cell Radius
	let sphere_rad = 600.0; //Neural Network Radius

	let mut netdata: raybnn::neural::network_f32::network_metadata_type = raybnn::neural::network_f32::network_metadata_type {
		neuron_size: neuron_size,
	    input_size: input_size,
		output_size: output_size,
		proc_num: proc_num,
		active_size: active_size,
		space_dims: space_dims,
		step_num: sim_steps,
		batch_size: batch_size,
		del_unused_neuron: true,

		time_step: 0.3,
		nratio: 0.5,
		neuron_std: 0.3,
		sphere_rad: sphere_rad,
		neuron_rad: neuron_rad,
		con_rad: 0.6,
		init_prob: 0.5,
		add_neuron_rate: 0.0,
		del_neuron_rate: 0.0,
		center_const: 0.005,
		spring_const: 0.01,
		repel_const: 0.01
	};

	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	arrayfire::sync(DEVICE);

	let mut final_neuron_num = Vec::new();
	let mut final_glia_num = Vec::new();

	let mut collision_run_time = Vec::new();


	let mut initial_cell_num = active_size_list.clone();
	initial_cell_num.iter_mut().for_each(|b| *b = (*b * 2));

	println!("Initial Number of Cells: {:?}", initial_cell_num);

	


	//Loop Through Different Neural Network Sizes
	arrayfire::sync(DEVICE);
	for ii in 0..10
	{
		for new_active_size in active_size_list.clone()
		{
			netdata.active_size = new_active_size.clone();

			//println!("new_active_size {} i {}",new_active_size, ii);
	
			let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
			let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);
		
			arrayfire::sync(DEVICE);
			let start = Instant::now();
	
			//Create Neural Network Sphere
			raybnn::physics::initial_f32::sphere_cell_collision_serial(
				&netdata,
				&mut glia_pos,
				&mut neuron_pos
			);
		
			//Record the number of neural network cells

			final_neuron_num.push(neuron_pos.dims()[0]);
			final_glia_num.push(glia_pos.dims()[0]);
	
			let duration = start.elapsed();
	
			collision_run_time.push(duration.as_secs_f64());
	
		}
		std::thread::sleep(std::time::Duration::from_secs(10));
	}

	//Print the final number of neural network cells after deleting collided cells
    
	println!("final_neuron_num {:?}",final_neuron_num);
	println!("final_glia_num {:?}",final_glia_num);
	println!("collision_run_time {:?}",collision_run_time);


	let mut collision_run_time_f64 = Vec::new();
	for element in collision_run_time.clone() { 
		collision_run_time_f64.push(element as f64);
	}
	//Save Collision Runtime
	raybnn::export::dataloader_f64::write_vec_cpu_to_csv(
		"./collision_run_time_serial.csv",
		&collision_run_time_f64
	);



}
