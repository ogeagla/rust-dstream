extern crate nalgebra as na;

use dstream::dstream2d::*;
use dstream::dstreamn::*;
use na::*;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;
use petgraph::visit::*;
use std::collections::HashMap;

fn pretty_print_dmat(dmat: DMatrix<f64>) {
    for r in 0..dmat.nrows() {
        for c in 0..dmat.ncols() {
            let elem = dmat[(r, c)];
            print!("{}, ", elem);
        }
        println!(" ");
    }
    println!("");
}

#[test]
fn dstreamn () {
    println!("****** DSTREAMN START");
    do_stuff();
    println!("****** DSTREAMN END");
}


#[test]
fn test_graph_connected() {
    let mut g = Graph::<(usize, usize), (usize, usize)>::new();

    let n1 = g.add_node((0,0));
    let n2 = g.add_node((0,1));
    let n3 = g.add_node((1,0));

    g.extend_with_edges(&[(n1, n2)]);
    g.extend_with_edges(&[(n1, n3)]);

    assert_eq!(true, TheWorld::graph_is_fully_connected(g));
}


#[test]
fn test_graph_disconnected() {
    let mut g = Graph::<(usize, usize), (usize, usize)>::new();

    let n1 = g.add_node((0,0));
    let n2 = g.add_node((0,1));
    let n3 = g.add_node((1,0));
    let n4 = g.add_node((1,4));
    let n5 = g.add_node((1,5));

    g.extend_with_edges(&[(n1, n2)]);
    g.extend_with_edges(&[(n1, n3)]);
    g.extend_with_edges(&[(n4, n5)]);

    assert_eq!(false, TheWorld::graph_is_fully_connected(g));
}

#[test]
#[ignore]
fn test_mark_and_remove_and_reset_spore_adics() {
    //TODO
    assert!(false);
}

#[test]
#[ignore]
fn test_is_a_grid_cluster() {
    //TODO
    assert!(false);
}

#[test]
fn test_get_neighbors() {
    let dg1 = GridCell {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg2 = GridCell {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg3 = GridCell {i: 0, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg4 = GridCell {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg5 = GridCell {i: 1, j: 4,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg6 = GridCell {i: 1, j: 2,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};

    let neighbors1 = TheWorld::get_neighbors(dg4.clone(), vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()));
    let neighbors2 = TheWorld::get_neighbors(dg5.clone(), vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()));
    assert_eq!(3, neighbors1.len());
    assert_eq!(0, neighbors2.len());
}

#[test]
fn test_is_inside_grid() {
    let dg1 = GridCell {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg2 = GridCell {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg3 = GridCell {i: 0, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg4 = GridCell {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg5 = GridCell {i: 1, j: 4,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg6 = GridCell {i: 1, j: 2,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};

    let result_1 = TheWorld::is_inside_grid(dg1.clone(), vec![dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()]);
    let result_2 = TheWorld::is_inside_grid(dg6.clone(), vec![dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()]);

    assert_eq!(true, result_1);
    assert_eq!(false, result_2);
}

#[test]
fn test_is_a_grid_group() {
    let dg1 = GridCell {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg2 = GridCell {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg3 = GridCell {i: 0, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg4 = GridCell {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg5 = GridCell {i: 1, j: 4,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg6 = GridCell {i: 1, j: 2,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};

    let result = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg6.clone()));
    assert_eq!(true, result);

    let result2 = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone()));
    assert_eq!(false, result2);

    let result3 = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(),));
    assert_eq!(true, result3);
}

#[test]
fn test_compute_grid_indxs() {
    let loc = (-6., 6.);
    let i_rn = (-10.0, 10.0);
    let j_rn = (-10.0, 10.0);
    let i_bins = 10 as usize;
    let j_bins = 10 as usize;

    let result = TheWorld::compute_grid_indxs(loc, i_rn, j_rn, i_bins, j_bins).unwrap();
    assert_eq!((2, 8), result);

}

#[test]
fn test_are_neighbors() {
    let dg1 = &GridCell {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg2 = &GridCell {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg3 = &GridCell {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};

    assert_eq!(true, TheWorld::are_neighbors(dg1, dg2));
    assert_eq!(true, TheWorld::are_neighbors(dg2, dg3));
    assert_eq!(false, TheWorld::are_neighbors(dg1, dg3));
}

#[test]
fn test_dg_update_and_get() {
    let mut dg = GridCell {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    dg.update(1, vec!(100.0));

    dg.update(10, vec!(200.0));
    dg.update(20, vec!(300.0));

    assert_eq!((1, 100.0), dg.get_last_update_and_value_to(2));
    assert_eq!((10, 200.0), dg.get_last_update_and_value_to(14));
    assert_eq!((20, 300.0), dg.get_last_update_and_value_to(22));

    let v_at_t = dg.get_at_time(30);
    println!("t: {}, v: {}", v_at_t.0, v_at_t.1);
}

#[test]
fn test_removed_as_sporadic() {
    let mut dg = GridCell {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: vec!(1, 2, 3, 4), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    assert_eq!(1, dg.get_last_time_removed_as_sporadic_to(2));
    assert_eq!(4, dg.get_last_time_removed_as_sporadic_to(6));
}

#[test]
fn test_put_time_updates_work() {
    let raw_data_1 = RawData{x: 1.0, y: -1.0, v: 123.45};
    let raw_data_2 = RawData{x: 5.0, y: -7.0, v: 23.45};
    let raw_data_3 = RawData{x: 5.0, y: -7.0, v: 83.45};
    let raw_data_4 = RawData{x: 15.0, y: -17.0, v: 83.45};

    let rd_vec = vec!(raw_data_1, raw_data_2, raw_data_3, raw_data_4);

    let t = 1;

    let default_vec : Vec<GridPoint> = Vec::new();
    let mut world = TheWorld{grid_cells: Vec::new(), timeline: Vec::new(),
        current_time: 0, clusters: Vec::new(), previous_grid_cells: Vec::new(),};
    world.init(default_vec);
    let res1 = world.put(t, rd_vec.clone());
    assert_eq!(t, world.current_time);
    let res2 = world.put(t + 1, rd_vec.clone());
    assert_eq!(t + 1, world.current_time);
    let res3 = world.put(t + 2, rd_vec.clone());
    assert_eq!(t + 2, world.current_time);
    assert_eq!(vec!(t, t+1, t+2), world.timeline);
}

#[test]
fn runs() {
    println!("Testing the world...");
    Runner::run_world();
    println!("Testing the world complete.");
}

#[test]
fn test_labels_changed_between() {

    let mut map1: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map1.insert((0, 1), GridLabel::Dense);
    map1.insert((1, 1), GridLabel::Transitional);

    let mut map2: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map2.insert((0, 1), GridLabel::Dense);
    map2.insert((1, 1), GridLabel::Transitional);

    let mut map3: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map3.insert((0, 1), GridLabel::Dense);
    map3.insert((1, 1), GridLabel::Sparse);

    assert_eq!(false, TheWorld::labels_changed_between(map1.clone(), map2.clone()));
    assert_eq!(false, TheWorld::labels_changed_between(map2.clone(), map1.clone()));
    assert_eq!(true, TheWorld::labels_changed_between(map1.clone(), map3.clone()));
    assert_eq!(true, TheWorld::labels_changed_between(map3.clone(), map1.clone()));

}

#[test]
fn test_which_labels_changed_between() {

    let mut map1: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map1.insert((0, 1), GridLabel::Dense);
    map1.insert((1, 1), GridLabel::Transitional);

    let mut map2: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map2.insert((0, 1), GridLabel::Dense);
    map2.insert((1, 1), GridLabel::Transitional);

    let mut map3: HashMap<(usize, usize), GridLabel> = HashMap::new();
    map3.insert((0, 1), GridLabel::Dense);
    map3.insert((1, 1), GridLabel::Sparse);

    assert_eq!(None, TheWorld::which_labels_changed_between(map1.clone(), map2.clone()));
    assert_eq!(None, TheWorld::which_labels_changed_between(map2.clone(), map1.clone()));
    assert_eq!(Some(vec!((1,1))), TheWorld::which_labels_changed_between(map1.clone(), map3.clone()));
    assert_eq!(Some(vec!((1,1))), TheWorld::which_labels_changed_between(map3.clone(), map1.clone()));
}

#[test]
fn test_is_outside_when_added_to() {

    let dg1 = GridCell {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg2 = GridCell {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg3 = GridCell {i: 0, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg4 = GridCell {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg5 = GridCell {i: 1, j: 4,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
    let dg6 = GridCell {i: 1, j: 2,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};

    let result_1 = TheWorld::is_outside_when_added_to(
        dg1.clone(),
        dg2.clone(),
        vec![dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()]);
    let result_2 = TheWorld::is_outside_when_added_to(
        dg6.clone(),
        dg2.clone(),
        vec![dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone(), dg6.clone()]);

    assert_eq!(false, result_1);
    assert_eq!(true, result_2);
}
