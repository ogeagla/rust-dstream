use na::{DMatrix};
use std::collections::{HashMap, HashSet};
use std::num::*;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;
use rand;


pub fn do_stuff() {
    let d = 7; //dimensions
    let bin_width = 0.2; //bin size (assuming bins are squares
    let bin_range_start = -100.0; //bin values cover -100
    let bin_range_end = 100.0; //to 100
    let bin_range = rand::distributions::Range::new(bin_range_start, bin_range_end);
    let gap_time = 5; //time steps to run before adjusting clustering
    let c_m = 3.0;
    let c_l = 0.8;
    let lambda = 0.998;
    let beta = 0.3;

    let mut rng = rand::thread_rng();

    for t in 0..11 {


        let rand_vec = rand::sample(&mut rng, -100..100, d);
        println!("{:?}", rand_vec);

    }
}

