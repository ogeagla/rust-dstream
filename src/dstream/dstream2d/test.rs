extern crate nalgebra as na;

use dstream::dstream2d::{TheWorld, DG, RawData, GridPoint};
use na::*;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;
use petgraph::visit::*;

#[test]
fn test_graph_works_as_expected() {
    println!("-----graph behavior");
    let mut g = Graph::<(usize, usize), (usize, usize)>::new();
    
    let n1 = g.add_node((0,0));
    let n2 = g.add_node((0,1));
    let n3 = g.add_node((1,0));
    let n4 = g.add_node((1,4));
    let n5 = g.add_node((1,5));

    g.extend_with_edges(&[(n1, n2)]);
    g.extend_with_edges(&[(n1, n3)]);
    g.extend_with_edges(&[(n4, n5)]);

    let edge_count = g.edge_count();
    let node_count = g.node_count();

    println!("test edge count: {}", edge_count);
    println!("test node count: {}", node_count);

    for n in g.node_indices() {
        let nsu = g.neighbors_undirected(n);
        let esu = g.edges(n);
        println!("for node {}", n.index());
        for nu in nsu {
            println!("-- neighbor visited");
        }
    }

    //http://math.stackexchange.com/questions/551889/checking-if-a-graph-is-fully-connected
    println!("for graph with n nodes, then A^n, where A the adj matrix, should have no zero elements");
    let adj_mat = g.adjacency_matrix();

    let dim = 5;

    let dmat = convert_graph_adj_mat_to_nalgebra_mat(g, dim, dim);
    println!("the adj mat: ");
    pretty_print_dmat(dmat.clone());
    let mut dmat_powered = dmat.clone();
    for d in 0..(dim-1) {
        dmat_powered = dmat_powered.clone() * dmat.clone();
    }
    println!("checking neighbors; mat to the power of nodes:");
    pretty_print_dmat(dmat_powered.clone());
    println!("fully connected: {}", dmat_represents_fully_connected(dmat_powered));
    println!("----end graph behavior");

    fn dmat_represents_fully_connected(dmat: DMat<f64>) -> bool {
        for r in 0..dmat.nrows() {
            for c in 0..dmat.ncols() {
                let elem = dmat[(r, c)];
                if elem == 0.0 {
                    return false
                }
            }
        }
        true
    }

    fn pretty_print_dmat(dmat: DMat<f64>) {
        for r in 0..dmat.nrows() {
            for c in 0..dmat.ncols() {
                let elem = dmat[(r, c)];
                print!("{} ", elem);
            }
            println!(" ,");
        }
    }

    fn convert_graph_adj_mat_to_nalgebra_mat(g: Graph<(usize, usize), (usize, usize)>, rows: usize, cols: usize) -> DMat<f64> {

        let mut m2 : DMat<f64> = DMat::new_zeros(rows, cols);
        let count = rows * rows;
        let mut node_count = 0;

        for n1 in g.node_indices() {
            let mut neighs_v = Vec::new();
            let neighs = g.neighbors_undirected(n1);
            //TODO ugly way to covert iterator to vector
            for neigh in neighs {
                neighs_v.push(neigh);
            }
            let idx_1 = n1.index();
            for n2 in g.node_indices() {
                node_count += 1;
                let idx_2 = n2.index();
                println!("updating matrix for ({}, {})", idx_1, idx_2);
                if ! neighs_v.contains(&n2) {
                    (&mut m2)[(idx_1,idx_2)] = 0.0;

                } else {
                    (&mut m2)[(idx_1,idx_2)] = 1.0;
                }

                if idx_1 == idx_2 {
                    (&mut m2)[(idx_1,idx_2)] = 1.0;
                }
            }
        }
        assert_eq!(count, node_count);
        println!("rows: {}, cols: {}", m2.nrows(), m2.ncols());
        m2
    }
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
#[ignore]
fn test_is_inside_grid() {
    //TODO
    assert!(false);
}

#[test]
#[ignore]
fn test_is_a_grid_group() {
    //TODO
    let dg1 = DG {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg2 = DG {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg3 = DG {i: 0, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg4 = DG {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg5 = DG {i: 1, j: 4,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg6 = DG {i: 1, j: 2,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};

    let result = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg6.clone()));
    assert_eq!(true, result);

    let result2 = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(), dg4.clone(), dg5.clone()));
    assert_eq!(false, result2);

    let result3 = TheWorld::is_a_grid_group(vec!(dg1.clone(), dg2.clone(), dg3.clone(),));
    assert_eq!(true, result3);
}

#[test]
#[ignore]
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
#[ignore]
fn test_are_neighbors() {
    let dg1 = &DG {i: 0, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg2 = &DG {i: 1, j: 0,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    let dg3 = &DG {i: 1, j: 1,
        updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};

    assert_eq!(true, TheWorld::are_neighbors(dg1, dg2));
    assert_eq!(true, TheWorld::are_neighbors(dg2, dg3));
    assert_eq!(false, TheWorld::are_neighbors(dg1, dg3));
}

#[test]
#[ignore]
fn test_dg_update_and_get() {
    let mut dg = DG {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
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
#[ignore]
fn test_removed_as_sporadic() {
    let mut dg = DG {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: vec!(1, 2, 3, 4),};
    assert_eq!(1, dg.get_last_time_removed_as_sporadic_to(2));
    assert_eq!(4, dg.get_last_time_removed_as_sporadic_to(6));
}

#[test]
#[ignore]
fn test_put_works() {
    let raw_data_1 = RawData{x: 1.0, y: -1.0, v: 123.45};
    let raw_data_2 = RawData{x: 5.0, y: -7.0, v: 23.45};
    let raw_data_3 = RawData{x: 5.0, y: -7.0, v: 83.45};
    let raw_data_4 = RawData{x: 15.0, y: -17.0, v: 83.45};

    let rd_vec = vec!(raw_data_1, raw_data_2, raw_data_3, raw_data_4);

    let t = 1;

    let default_vec : Vec<GridPoint> = Vec::new();
    let mut world = TheWorld{g_vec: Vec::new()};
    world.init(default_vec);
    let res1 = world.put(t, rd_vec.clone());
    let res2 = world.put(t + 1, rd_vec.clone());
    let res3 = world.put(t + 2, rd_vec.clone());

}