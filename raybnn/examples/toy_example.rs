/* 
This is raybnn instance with a toy example from figure3a.rs

*/

extern crate arrayfire;
extern crate raybnn;
use std::collections::HashMap;
use nohash_hasher;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

use raybnn::physics::update_f32::add_neuron_option_type;

use arrayfire::af_print;

//use rand::SeedableRng;
//use rand::rngs::StdRng;

#[allow(unused_must_use)]
fn main() {
    
    // Raytracing 3 radius vector: vector contains the radius values of neurons associated with RT3 
    let RT3_RAD_vec: Vec<f32> = vec! [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
     90.0, 100.0, 110.0, 120.0, 130.0, 150.0, 200.0];

    // Save
    raybnn::export::dataloader_f32::write_vec_cpu_to_csv(
        "./RT3_RAD_vec.csv",
        &RT3_RAD_vec
    );


    // Set CUDA and GPU device 0
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


        //set seed
    //arrayfire::set_seed(42);

    // Run the code 1 times instead of 11 times 
    for run_num in 0..1
    {

        arrayfire::sync(DEVICE); /* Synchronizes the device GPU, wait for all AF operations on DEVICE 
        to complete before moving on */
        arrayfire::device_gc(); /* garbage collector for device, frees up unused memory that was 
        allocated by AF but is no longer being used */
        arrayfire::sync(DEVICE);

        
        //arrayfire::set_seed(42);
        let RT3_RAD_vec: Vec<f32> = vec! [10.0];
        //let RT3_RAD_vec: Vec<f32> = vec! [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0, 200.0];


        for RT3_RAD in RT3_RAD_vec.clone()
        {

        arrayfire::sync(DEVICE);
        arrayfire::device_gc();
        arrayfire::sync(DEVICE);

        // set seed
        //arrayfire::set_seed(42);        
        //arrayfire::set_seed((RT3_RAD as u64)*run_num);


        // number of connetions (synapses) per neuron
        let conn_vec: Vec<u64> =        vec![ 10,    10,    10,     50,     60,     20,   20,    20,    20,     20,    10];

        // vector contain different total number of neurons
        let neuron_size_vec: Vec<u64> = vec![ 40,    50,    70,    160,     170,   180,  190,   200,   230,    270,   300];

        // input vector for the network (number of features, nodes or input dimensions)
        let arg_input_vec: Vec<u64> = vec! [4];

        let dir_path = "/tmp/".to_string();

        // Set more NN Parameters

        let max_input_size: u64 = 4;
        let mut input_size: u64 = 4;

        let max_output_size: u64 = 3;
        let output_size: u64 = 3;

        let max_neuron_size: u64 = 10;

        let mut batch_size: u64 = 5;
        let mut traj_size: u64 = 1;     // number of time steps or neurons in a time step that are being processed

        let proc_num = 2;

        let train_size: u64 = 10;
        let test_size: u64 = 5;    // OK

        let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);

        // set seed
        arrayfire::set_seed(42);

        // Create Start Neural Network
    	println!("Creating Network");

        let mut arch_search = raybnn::interface::automatic_f32::create_start_archtecture(

            input_size,
            max_input_size,

            output_size,
            max_output_size,

            max_neuron_size,

            batch_size,
            traj_size,

            &dir_path
        );

        arch_search.neural_network.netdata.proc_num = proc_num;

        // Load dataset
        println!("Load Dataset");

        let TOTAL_RSSI_TESTX = raybnn::export::dataloader_f32::file_to_hash_cpu(
            "./test_data/testX.dat",
            max_input_size,
		    batch_size*traj_size
        );

        let RSSI_TESTY = raybnn::export::dataloader_f32::file_to_hash_cpu(
		"./test_data/testY.dat",
		output_size,
		batch_size*traj_size
	    );
        arrayfire::sync(DEVICE);
	    arrayfire::device_gc();
	    arrayfire::sync(DEVICE);

        let mut RSSI_TESTX = TOTAL_RSSI_TESTX.clone();

        let TOTAL_RSSI_TRAINX = raybnn::export::dataloader_f32::file_to_hash_cpu(
    	"./test_data/trainX.dat",
    	max_input_size,
		batch_size*traj_size
        );

        let RSSI_TRAINY = raybnn::export::dataloader_f32::file_to_hash_cpu(
    	"./test_data/trainY.dat",
    	output_size,
		batch_size*traj_size
        );
        arrayfire::sync(DEVICE);
	    arrayfire::device_gc();
	    arrayfire::sync(DEVICE);

        let mut RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();
        println!("RSSI_TRAINX: {:?}", RSSI_TRAINX);
        println!("RSSI_TRAINY: {:?}", RSSI_TRAINY);
        println!("RSSI_TESTX: {:?}", RSSI_TESTX);
        println!("RSSI_TESTY: {:?}", RSSI_TESTY);

        // === SAFETY CHECKS AND DEBUG PRINTS ===
        // 2. Check dataset size for batch settings
        let expected_train_len = (max_input_size * batch_size * traj_size) as usize;
        let expected_test_len = (max_input_size * batch_size * traj_size) as usize;
        
         println!("\n[DEBUG] RSSI_TRAINX keys: {:?}", RSSI_TRAINX.keys());
        for (key, value) in &RSSI_TRAINX {
            println!("[DEBUG] RSSI_TRAINX[{}] len = {} (expected {})", key, value.len(), expected_train_len);
            if value.len() != expected_train_len {
                println!("[WARNING] RSSI_TRAINX[{}] has length {}, expected {}", key, value.len(), expected_train_len);
            }
        }
        println!("[DEBUG] RSSI_TRAINY keys: {:?}", RSSI_TRAINY.keys());
        for (key, value) in &RSSI_TRAINY {
            println!("[DEBUG] RSSI_TRAINY[{}] len = {}", key, value.len());
        }
        println!("[DEBUG] RSSI_TESTX keys: {:?}", RSSI_TESTX.keys());
        for (key, value) in &RSSI_TESTX {
            println!("[DEBUG] RSSI_TESTX[{}] len = {} (expected {})", key, value.len(), expected_test_len);
            if value.len() != expected_test_len {
                println!("[WARNING] RSSI_TESTX[{}] has length {}, expected {}", key, value.len(), expected_test_len);
            }
        }
        println!("[DEBUG] RSSI_TESTY keys: {:?}", RSSI_TESTY.keys());
        for (key, value) in &RSSI_TESTY {
            println!("[DEBUG] RSSI_TESTY[{}] len = {}", key, value.len());
        }
        // Print warnings for missing keys in X or Y
        for k in RSSI_TRAINX.keys() {
            if !RSSI_TRAINY.contains_key(k) {
                println!("[WARNING] Key {} in TRAINX but not in TRAINY", k);
            }
        }
        for k in RSSI_TRAINY.keys() {
            if !RSSI_TRAINX.contains_key(k) {
                println!("[WARNING] Key {} in TRAINY but not in TRAINX", k);
            }
        }
        for k in RSSI_TESTX.keys() {
            if !RSSI_TESTY.contains_key(k) {
                println!("[WARNING] Key {} in TESTX but not in TESTY", k);
            }
        }
        for k in RSSI_TESTY.keys() {
            if !RSSI_TESTX.contains_key(k) {
                println!("[WARNING] Key {} in TESTY but not in TESTX", k);
            }
        }
        // === END SAFETY CHECKS ===
        
        // Transfer Learning training loop

        let mut prev_target_neuron_size = 0;

        for ii in 0..arg_input_vec.len()
        {
            RSSI_TRAINX = TOTAL_RSSI_TRAINX.clone();

            RSSI_TESTX = TOTAL_RSSI_TESTX.clone();

            arch_search.neural_network.neuron_pos = arrayfire::rows(&arch_search.neural_network.neuron_pos, input_size as i64, 
                (arch_search.neural_network.neuron_pos.dims()[0]-1) as i64);

            
            input_size = arg_input_vec[ii].clone();
            arch_search.neural_network.netdata.input_size = input_size;

            // need more clarification (conversion from CSR to COO matrix)
            let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

            arch_search.neural_network.netdata.neuron_std = 0.001;
            arch_search.neural_network.netdata.con_rad = (arch_search.neural_network.netdata.sphere_rad/(proc_num as f32))*2.0;

            // Initialize input neurons maybe with different radius

            let input_neurons = raybnn::physics::initial_f32::create_spaced_input_neuron_on_sphere_1D(
                arch_search.neural_network.netdata.sphere_rad+0.2,
                input_size
            );

            // added those input neurons back to neuron_pos
            arch_search.neural_network.neuron_pos = arrayfire::join(0, &input_neurons, &arch_search.neural_network.neuron_pos);

            // assigned active_size with new values including input_neurons
            arch_search.neural_network.netdata.active_size = arch_search.neural_network.neuron_pos.dims()[0];
            
            // creates gaps in the index space for future expansion -> when we add more input/output neurons later
            // they can use reserved indices
            raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
                max_input_size,
                max_output_size,
                &arch_search.neural_network.netdata,
                &arch_search.neural_network.neuron_pos,
                &mut arch_search.neural_network.neuron_idx
            );

            // Modify Input Dimensions

            if (prev_target_neuron_size != neuron_size_vec[ii])
            {
            // Calculate how many new hidden neurons need to be added to reach the target network size
            // new_neuron = total_num_neuron - (total_current_neuron-input_neuron-output_neurons) (current hidden neuron)    
            let new_neuron_num = neuron_size_vec[ii] - (arch_search.neural_network.neuron_idx.dims()[0] - input_size - output_size);


                prev_target_neuron_size = neuron_size_vec[ii];

                let add_neuron_options: add_neuron_option_type = add_neuron_option_type {
                    new_active_size: new_neuron_num,
                    init_connection_num: conn_vec[ii as usize].clone(),
                    input_neuron_con_rad: 40.0*arch_search.neural_network.netdata.neuron_rad,
                    hidden_neuron_con_rad: RT3_RAD*arch_search.neural_network.netdata.neuron_rad,
                    output_neuron_con_rad: 40.0*arch_search.neural_network.netdata.neuron_rad,
                };

                // Then add these new hidden neurons to existing sphere
                raybnn::physics::update_f32::add_neuron_to_existing3(
                    &add_neuron_options,
				
				    &mut arch_search,
                );
            }

            // call the function one more time because the number of hidden neurons changed-> need to recalculate the reserved neuron indexes 
            raybnn::physics::initial_f32::assign_neuron_idx_with_buffer(
			max_input_size,
			max_output_size,
			&arch_search.neural_network.netdata,
			&arch_search.neural_network.neuron_pos,
			&mut arch_search.neural_network.neuron_idx,
		);

            let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
            arch_search.neural_network.netdata.active_size = active_size;

            WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);

            arch_search.neural_network.WRowIdxCSR = raybnn::graph::large_sparse_i32::COO_to_CSR(&WRowIdxCOO, arch_search.neural_network.netdata.neuron_size);

            arrayfire::device_gc();
            arrayfire::sync(DEVICE);


            // Modify training dataset (this code resizes every sample in dataset to match the required input dimension)

            if (input_size != max_input_size)
            {
                // clone dataset
                let mut tempRSSI = TOTAL_RSSI_TRAINX.clone();

                for (key, value) in &TOTAL_RSSI_TRAINX {
                    let mut tempvec = value.clone();
                    
                    // prepare data for arrayfire
                    let veclen = tempvec.len() as u64;

                    // creates an ArrayFire array with a shape determined by the maximum input size
                    let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[max_input_size, veclen/max_input_size, 1, 1]));

                    // select only the first input_size rows from AF array, reducing feature dimension
                    formatarr = arrayfire::rows(&formatarr, 0, (input_size -1) as i64);

                    // Creates a new vector tempvec filled with default f32 values (usually 0.0), sized to match the reformatted AF array.
                    tempvec = vec!(f32::default(); formatarr.elements());

                    // copies the data from GPU/AF back into the Rust vector.
                    formatarr.host(&mut tempvec);

                    // Inserts the new, resized vector back into the temp dataset under the same key.
                    tempRSSI.insert(key.clone(), tempvec);
                }

            }

                arrayfire::sync(DEVICE);

                // this could be similar to eq: U=It+k in the paper
                let traj_steps = traj_size + proc_num - 1;

                // Internal state pre-activation Z to store values for every neuron, sample in the batch, step in the trajectory
                let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size, batch_size, traj_steps, 1]);
                let mut Z = arrayfire::constant::<f32>(0.0, Z_dims);

                // Internal state post-activation matrix or temporary output Q 
                let mut Q = arrayfire::constant::<f32>(0.0, Z_dims);


                let mut alpha0: f32 = 0.0001;

                let mut active_size = arch_search.neural_network.neuron_idx.dims()[0];
                // extract output neurons for indexing
                let idxsel = arrayfire::rows(&arch_search.neural_network.neuron_idx, 
                    (active_size - output_size) as i64, (active_size - 1) as i64);

                let Qslices: u64 = Q.dims()[2]; //= traj_step = 2

                let mut WRowIdxCOO = raybnn::graph::large_sparse_i32::CSR_to_COO(&arch_search.neural_network.WRowIdxCSR);



                let total_param_size = arch_search.neural_network.network_params.dims()[0];
                
                // params used for Adam optimizer
                let mt_dims = arrayfire::Dim4::new(&[total_param_size,1,1,1]);
                let mut mt = arrayfire::constant::<f32>(0.0,mt_dims);
                let mut vt = arrayfire::constant::<f32>(0.0,mt_dims);
                let mut grad = arrayfire::constant::<f32>(0.0,mt_dims);

                // Define Sequence arrays used to index and extract specific parameter ranges
                let mut Wseqs = [arrayfire::Seq::default()];
                let mut Hseqs = [arrayfire::Seq::default()];
                let mut Aseqs = [arrayfire::Seq::default()];
                let mut Bseqs = [arrayfire::Seq::default()];
                let mut Cseqs = [arrayfire::Seq::default()];
                let mut Dseqs = [arrayfire::Seq::default()];
                let mut Eseqs = [arrayfire::Seq::default()];


                let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
		        let mut X = arrayfire::constant::<f32>(0.0,X_dims);

                // Define all necessary params used for forward pass and backward pass
                let mut idxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut valsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

                let mut cvec_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut dXsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();

                let mut nrows_out: nohash_hasher::IntMap<i64, u64> = nohash_hasher::IntMap::default();
                let mut sparseval_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut sparsecol_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut sparserow_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();




                let mut Hidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut Aidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut Bidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut Cidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut Didxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut Eidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();
                let mut combidxsel_out: nohash_hasher::IntMap<i64, arrayfire::Array<i32> > = nohash_hasher::IntMap::default();


                let mut dAseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
                let mut dBseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
                let mut dCseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
                let mut dDseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
                let mut dEseqs_out: nohash_hasher::IntMap<i64, [arrayfire::Seq<i32>; 2] > = nohash_hasher::IntMap::default();
                
                
                let WValuesdims0 =  arch_search.neural_network.WColIdx.dims()[0];

                let network_paramsdims0 =  arch_search.neural_network.network_params.dims()[0];

                let Hdims0 =  (network_paramsdims0 -  WValuesdims0)/6; 

                // Create backward graph
                
                raybnn::graph::path_f32::find_path_backward_group2(
                    &arch_search.neural_network.netdata,
                    traj_steps,
                    traj_size, 
                    &WRowIdxCOO,
                    &arch_search.neural_network.WColIdx,
                    &arch_search.neural_network.neuron_idx,

                    WValuesdims0,
                    Hdims0,
                    Hdims0,
                    Hdims0,
                    Hdims0,
                    Hdims0,
                    Hdims0,
                
                
                    
                    &mut idxsel_out,
                    &mut valsel_out,

                    &mut cvec_out,
                    &mut dXsel_out,

                    &mut nrows_out,
                    &mut sparseval_out,
                    &mut sparserow_out,
                    &mut sparsecol_out,
                    
                
                
                    &mut Hidxsel_out,
                    &mut Aidxsel_out,
                    &mut Bidxsel_out,
                    &mut Cidxsel_out,
                    &mut Didxsel_out,
                    &mut Eidxsel_out,
                    &mut combidxsel_out,
                
                
                
                

                    &mut dAseqs_out,
                    &mut dBseqs_out,
                    &mut dCseqs_out,
                    &mut dDseqs_out,
                    &mut dEseqs_out,
                
                
                
                    
                    &mut Wseqs,
                    &mut Hseqs,
                    &mut Aseqs,
                    &mut Bseqs,
                    &mut Cseqs,
                    &mut Dseqs,
                    &mut Eseqs
                );

            
            let train_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
            let mut train_X =  arrayfire::constant::<f32>(0.0,temp_dims);
            let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
            let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
            let mut batch_idx = 0;
            let epoch_num = (train_size/ (batch_size*traj_size) )-1; //2
            println!("epoch_num: {:?}", epoch_num);
            
            let mut max_epoch = 1;


            let mut loss_val = 100000.0;
            let mut loss_val2 = 100000.0;
		    let mut prev_loss = 100000.0;

            arrayfire::device_gc();
		    arrayfire::sync(DEVICE);

            let start_time = std::time::Instant::now();

            // Training loop

            for i in 0..max_epoch
            {                           //i=0,1,2,3,4,...,19
                                        //batch_idx = 0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5
                batch_idx = i% epoch_num; // 

                // Check key existence before accessing
                if !RSSI_TRAINX.contains_key(&batch_idx) {
                    panic!("Missing batch {} in RSSI_TRAINX!", batch_idx);
                }
                if !RSSI_TRAINY.contains_key(&batch_idx) {
                    panic!("Missing batch {} in RSSI_TRAINY!", batch_idx);
                }


                train_X = arrayfire::Array::new(&RSSI_TRAINX[&batch_idx], train_X_dims);
                println!("train_X_dims: {:?}", train_X_dims);
                println!("train_X: {:?}", train_X);
                Y = arrayfire::Array::new(&RSSI_TRAINY[&batch_idx], Y_dims);
                println!("Y_dims: {:?}", Y_dims);
                af_print!("Y: ", Y);
                arrayfire::set_slices(&mut X, &train_X, 0,(traj_size-1) as i64);
                af_print!("X: ", X);

                // Forward pass
                println!("----------  forward pass using train set  ----------");
                raybnn::neural::network_f32::state_space_forward_batch(
                    &arch_search.neural_network.netdata,
                    &X,
                    &arch_search.neural_network.WRowIdxCSR,
                    &arch_search.neural_network.WColIdx,

                    &Wseqs,
                    &Hseqs,
                    &Aseqs,
                    &Bseqs,
                    &Cseqs,
                    &Dseqs,
                    &Eseqs,
                    &arch_search.neural_network.network_params,

                    &mut Z,
				    &mut Q
                );


                // Get Yhat
                let mut idxrs = arrayfire::Indexer::default();

                // why need to have these sequences? used for slicing arrays in AF
                // seq1 is the sequence array object started from 0 to batch_size-1 with step=1
                let seq1 = arrayfire::Seq::new(0.0f32, (batch_size - 1) as f32, 1.0);

                // seq2 started from (proc_num-1) to (Qslices-1) with step=1
                // because proc_num-1 = 1 and Qslices = traj_step=2 --> 
                let seq2 = arrayfire::Seq::new((proc_num - 1) as f32, (Qslices - 1) as f32, 1.0);
                
                // assign idxsel, seq1 and seq2 to index slices (idxrs)
                idxrs.set_index(&idxsel, 0, None);
                idxrs.set_index(&seq1, 1, None);
                idxrs.set_index(&seq2, 2, None);
                
                // basically slices Q by idxrs
                let Yhat = arrayfire::index_gen(&Q, idxrs); 
                loss_val = raybnn::optimal::loss_f32::RMSE(&Yhat,&Y); // use softmax

                if i ==0
                {
                    loss_val2 = loss_val;
                }
                else
			    {
				loss_val2 = 0.99*loss_val2 + 0.01*loss_val;
			    }

                // Backward pass

                raybnn::neural::network_f32::state_space_backward_group2(
                    &arch_search.neural_network.netdata,                // NN Metadata
                    &X,                                                 // Input Matrix

                    &arch_search.neural_network.network_params,         // All trainable parameters in the network

                    &Z,                                                 // Internal state matrix Z
                    &Q,                                                 // Internal state matrix Q
                    &Y,                                                 // Target matrix to fit
                    raybnn::optimal::loss_f32::MSE_grad,                // loss function
                    &arch_search.neural_network.neuron_idx,             // Indexes of the neurons

                    &idxsel_out,                                        // Indexes of the UAF parameters
                    &valsel_out,                                        // Indexes of the UAF values

                    &cvec_out,                                          // Indexes of the UAF column sparse vector
                    &dXsel_out,                                         // Indexes of the dX values
                
                    &nrows_out,                                         // Number of rows in UAF
                    &sparseval_out,                                     // Indexes of the values in the sparse matrix
                    &sparserow_out,                                     // Indexes of the rows in the sparse matrix
                    &sparsecol_out,                                     // Indexes of the columns in the sparse matrix

                    &Hidxsel_out,                                       // Indexes of the bias vector H
                    &Aidxsel_out,                                       // Indexes of the UAF vector A
                    &Bidxsel_out,                                       // Indexes of the UAF vector B
                    &Cidxsel_out,                                       // Indexes of the UAF vector C
                    &Didxsel_out,                                       // Indexes of the UAF vector D
                    &Eidxsel_out,                                       // Indexes of the UAF vector E
                    &combidxsel_out,                                    // Indexes of the all vectors

                    &dAseqs_out,                                        // Indexes of the dA
                    &dBseqs_out,                                        // Indexes of the dB
                    &dCseqs_out,                                        // Indexes of the dC
                    &dDseqs_out,                                        // Indexes of the dD
                    &dEseqs_out,                                        // Indexes of the dE
			
                    // dL/dX:          Gradient of the loss function with respect to X
				    &mut grad,                                          // Gradient of all trainable parameters

                );

                //Update weights with ADAM
                                                                                                                                                                                        
                raybnn::optimal::gd_f32::adam(
                    0.9
                    ,0.999
                    ,&mut grad
                    ,&mut mt
                    ,&mut vt
                );

                alpha0 = (0.002/(loss_val + 1e-10)).min(0.007);

                if ii == 4 // original = 11
                {
                    alpha0 = alpha0.min(0.005);
                }

                arch_search.neural_network.network_params = arch_search.neural_network.network_params + (alpha0*-grad.clone());


                if (i>5) && ((i%2) == 0) // Changed
                {
                    if (prev_loss - loss_val).abs()/loss_val < 0.007
                    {
                        break;
                    }

                    prev_loss = loss_val;

                }

            }


            arrayfire::sync(DEVICE);
            println!("loss_val {}", loss_val);

            let elapsed: f32 = start_time.elapsed().as_secs_f32();
            println!("elapsed {}", elapsed);

            traj_size = 1;

            arrayfire::device_gc();

            // Change Input dimension of testing dataset

            if (input_size != max_input_size)
            {
                let mut tempRSSI = TOTAL_RSSI_TESTX.clone();

                for (key, value) in & TOTAL_RSSI_TESTX {
                    let mut tempvec = value.clone();

                    let veclen = tempvec.len() as u64;
                    let mut formatarr = arrayfire::Array::new(&tempvec, arrayfire::Dim4::new(&[
                        max_input_size, veclen/max_input_size, 1, 1]));

                    formatarr = arrayfire::rows(&formatarr, 0, (input_size-1) as i64);

                    tempvec = vec!(f32::default();formatarr.elements());
                    formatarr.host(&mut tempvec);

                    tempRSSI.insert(key.clone(), tempvec);
                }

                RSSI_TESTX = tempRSSI;

            }

            arrayfire::sync(DEVICE);


            let traj_steps = traj_size+proc_num-1;


            let Z_dims = arrayfire::Dim4::new(&[arch_search.neural_network.netdata.neuron_size,batch_size,traj_steps,1]);
            let mut Z = arrayfire::constant::<f32>(0.0,Z_dims);
            let mut Q = arrayfire::constant::<f32>(0.0,Z_dims);



            let Qslices: u64 = Q.dims()[2];


            let X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_steps,1]);
            let mut X = arrayfire::constant::<f32>(0.0,X_dims);



            let test_X_dims = arrayfire::Dim4::new(&[input_size,batch_size,traj_size,1]);
            let mut test_X =  arrayfire::constant::<f32>(0.0,temp_dims);
            let Y_dims = arrayfire::Dim4::new(&[output_size,batch_size,traj_size,1]);
            let mut Y =  arrayfire::constant::<f32>(0.0,temp_dims);
            let mut batch_idx = 0;
            let epoch_num = (test_size/ (batch_size*traj_size) );



            let temp_Y_dims = arrayfire::Dim4::new(&[1,output_size,1,1]);

            let mut Yhat_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);
            let mut Y_arr = arrayfire::constant::<f32>(0.0,temp_Y_dims);


            let newdims = arrayfire::Dim4::new(&[output_size,batch_size*traj_size,1,1]);


            //Compute testing dataset

            arrayfire::device_gc();
            arrayfire::sync(DEVICE);
            for batch_idx in 0..epoch_num
            {
                test_X =  arrayfire::Array::new(&RSSI_TESTX[&batch_idx], test_X_dims);

                Y = arrayfire::Array::new(&RSSI_TESTY[&batch_idx], Y_dims);
                arrayfire::set_slices(&mut X, &test_X, 0,(traj_size-1) as i64);


                                
                println!("----------  forward pass using test set  ----------");
                raybnn::neural::network_f32::state_space_forward_batch(
                    &arch_search.neural_network.netdata,
                    &X,
                    
                    &arch_search.neural_network.WRowIdxCSR,
                    &arch_search.neural_network.WColIdx,
                
                
                    &Wseqs,
                    &Hseqs,
                    &Aseqs,
                    &Bseqs,
                    &Cseqs,
                    &Dseqs,
                    &Eseqs,
                    &arch_search.neural_network.network_params,
                
                
                
                    &mut Z,
                    &mut Q
                );


                //Get Yhat
                let mut idxrs = arrayfire::Indexer::default();
                let seq1 = arrayfire::Seq::new(0.0f32, (batch_size-1) as f32, 1.0);
                let seq2 = arrayfire::Seq::new((proc_num-1) as f32, (Qslices-1) as f32, 1.0);
                idxrs.set_index(&idxsel, 0, None);
                idxrs.set_index(&seq1, 1, None);
                idxrs.set_index(&seq2, 2, None);
                let mut Yhat = arrayfire::index_gen(&Q, idxrs);
                


                Yhat = arrayfire::flat(&Yhat);
                Y = arrayfire::flat(&Y);


                Yhat = arrayfire::moddims(&Yhat, newdims);
                Y = arrayfire::moddims(&Y, newdims);


            
                Yhat = arrayfire::transpose(&Yhat, false);
                Y = arrayfire::transpose(&Y, false);
            

                Yhat_arr = arrayfire::join(0, &Yhat_arr, &Yhat);
                Y_arr = arrayfire::join(0, &Y_arr, &Y);
            }


            Yhat_arr = arrayfire::rows(&Yhat_arr,1,(Yhat_arr.dims()[0]-1) as i64);
            Y_arr = arrayfire::rows(&Y_arr,1,(Y_arr.dims()[0]-1) as i64);

            
            arrayfire::device_gc();
            arrayfire::sync(DEVICE);


            let totalparam = arch_search.neural_network.network_params.dims()[0] as f32;

            let v0: [f32; 2] = [elapsed, totalparam];
            let info = arrayfire::Array::new(&v0, arrayfire::Dim4::new(&[2, 1, 1, 1]));


            //Save dataset
            let save_filename = format!("toy_example_info_{}_{}_{}.csv",RT3_RAD as u64,input_size, run_num);
            raybnn::export::dataloader_f32::write_arr_to_csv(
                &save_filename,
                &info
            );

            

            let save_filename = format!("toy_example_test_pred_{}_{}_{}.csv",RT3_RAD as u64,input_size, run_num);
            raybnn::export::dataloader_f32::write_arr_to_csv(
                &save_filename,
                &Yhat_arr
            );



            let save_filename = format!("toy_example_test_act_{}_{}_{}.csv",RT3_RAD as u64,input_size, run_num);
            raybnn::export::dataloader_f32::write_arr_to_csv(
                &save_filename,
                &Y_arr
            );

        }

    }
}
println!("Done without any errors!");
}
