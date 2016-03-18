use na::{Mat2, DMat};
use std::collections::HashMap;
use std::num::*;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;
use rand;

mod test;

#[derive(Debug)]
#[derive(Clone)]
pub struct DG {
    i: usize,
    j: usize,
    updates_and_vals: Vec<GridPoint>,
    removed_as_spore_adic: Vec<u32>,
}

#[derive(Debug)]
#[derive(Clone)]
pub enum GridLabel { Dense, Sparse, Transitional, }

#[derive(Debug)]
#[derive(Clone)]
pub enum GridStatus { Sporadic, Normal, }

pub struct DStreamProps {
    c_m: f64,
    c_l: f64,
    lambda: f64,
    beta: f64,
    i_bins: usize,
    j_bins: usize,
    i_range: (f64, f64),
    j_range: (f64, f64),
    gap_time: u32,
}

#[derive(Clone)]
pub struct RawData {
    x: f64,
    y: f64,
    v: f64,
}

#[derive(Clone)]
#[derive(Debug)]
#[derive(Copy)]
pub struct GridPoint {
    t: u32,
    v: f64,
}
#[derive(Clone)]
pub struct GridData {
    i: usize,
    j: usize,
    v: f64,
}
pub struct TheWorld {
    g_vec: Vec<((usize, usize), DG)>,
    timeline: Vec<u32>,
    current_time: u32,
}

impl Default for DStreamProps {
    fn default() -> DStreamProps {
        DStreamProps {
            c_m: 3.0,
            c_l: 0.8,
            lambda: 0.998,
            beta: 0.3,
            i_bins: 100 as usize,
            j_bins: 100 as usize,
            i_range: (-10.0, 10.0),
            j_range: (-10.0, 10.0),
            gap_time: 5,
        }
    }
}

pub struct Runner;

impl Runner {
    pub fn run_world() {
        let props: DStreamProps = DStreamProps { ..Default::default() };

        let default_vec : Vec<GridPoint> = Vec::new();
        let mut world = TheWorld{g_vec: Vec::new(), timeline: Vec::new(), current_time: 0};
        world.init(default_vec);

        let mut has_initialized = false;
        for t in 0..11 {

            let r_x = rand::random::<f64>();
            let r_y = rand::random::<f64>();
            let v = rand::random::<f64>();

            let rd_1 = RawData { x: r_x, y: r_y, v: v};
            println!("putting rand raw data: ({}, {}) -> {}", r_x, r_y, v);
            let res = world.put(t, vec!(rd_1));

            if (t + 1) % props.gap_time == 0 {
                if has_initialized {
                    println!("-- adjusting clusters")
                } else {
                    println!("-- initializing clusters");
                    let result = world.initialize_clustering();
                    has_initialized = true
                }
            }
        }
    }
}

impl TheWorld {

    fn pretty_print_dmat(dmat: DMat<f64>) {
        for r in 0..dmat.nrows() {
            for c in 0..dmat.ncols() {
                let elem = dmat[(r, c)];
                print!("{}, ", elem);
            }
            println!(" ");
        }
        println!("");
    }

    fn get_labels_for_time(t: u32, dgs: Vec<DG>) -> HashMap<(usize, usize), GridLabel> {
        let mut the_map = HashMap::new();
        dgs.into_iter().map(|dg| {
            the_map.insert((dg.i, dg.j), dg.get_grid_label_at_time(t));
        });
        the_map
    }

    pub fn initialize_clustering(&mut self) -> Result<(), String> {
        //update density of all grids in grid_list
        //assign each dense grid to a distinct cluster
        //label all other grids as NO_CLASS; bad grids!
        /*
        do until no change in cluster labels
              foreach cluster c
                foreach outside grid g of c
                    foreach neighboring grid h of g
                        if h belongs to cluster c'
                            if |c| > |c'| label all grids in c' as c
                            else label all grids in c as c'
                        else if (h is translational) label h as in c
        end do
        */
        Ok(())
    }

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
        m2
    }

    ///
    ///   Here we take the adjacency matrix, `A`, and compute
    /// ```A' = A^d```
    /// where `d` is the number of nodes in the graph.
    ///   If any elements of `A'` are zero, then visitation has
    /// not occurred after `d` moves on the graph for some
    /// combinations, therefore graph is not fully connected,
    /// and return `false`.
    ///   This is because `d` is the most number of node hops
    /// required to visit all neighbors in a graph with node
    /// count `d`.
    ///
    pub fn graph_is_fully_connected(g: Graph<(usize, usize), (usize, usize)>) -> bool {
        let dim = g.node_count();
        let dmat = TheWorld::convert_graph_adj_mat_to_nalgebra_mat(g, dim, dim);

        let mut dmat_powered = dmat.clone();

        for d in 0..(dim-1) {
            dmat_powered = dmat_powered.clone() * dmat.clone();
        }
        TheWorld::pretty_print_dmat(dmat_powered.clone());
        TheWorld::dmat_represents_fully_connected(dmat_powered)
    }

    fn dgs_to_graph(dgs: Vec<DG>) ->  Graph<(usize, usize), (usize, usize)> {
        let mut neighbors_graph = Graph::<(usize, usize), (usize, usize)>::new();

        let dgs_and_nidxs: Vec<(DG, NodeIndex)> = dgs.clone().into_iter().map(|dg| (dg.clone(), neighbors_graph.add_node((dg.i, dg.j)))).collect();

        for two_dgs_and_nidxs in dgs_and_nidxs.clone().into_iter().combinations_n(2) {

            let dg1 = two_dgs_and_nidxs[0].0.clone();
            let dg2 = two_dgs_and_nidxs[1].0.clone();
            let idx1 = two_dgs_and_nidxs[0].1;
            let idx2 = two_dgs_and_nidxs[1].1;
            let i1 = dg1.i;
            let j1 = dg1.j;
            let i2 = dg2.i;
            let j2 = dg2.j;

            if TheWorld::are_neighbors(&dg1, &dg2) {
                neighbors_graph.extend_with_edges(&[(idx1, idx2)]);
            }
        }

        neighbors_graph.clone()
    }

    fn mark_and_remove_and_reset_spore_adics(t: u32, dgs: Vec<DG>) {
        //prop 4.3: mark as sporadic; grids marked sporadic last t can be removed this t, or labeled as normal
        //TODO
    }

    ///  if every inside grid of G is a dense grid and
    ///  every outside grid is either a dense grid or a transitional
    ///  grid, then G is a grid cluster.
    // TODO horrifying use of clone in this whole file...
    fn is_a_grid_cluster(t: u32, dgs: Vec<DG>) -> bool {
        for dg in dgs.clone().into_iter() {
            let label = dg.clone().get_grid_label_at_time(t);
            if TheWorld::is_inside_grid(dg.clone(), dgs.clone()) {
                match label {
                    GridLabel::Dense => (),
                    _ => return false,
                };
            } else {
                match label {
                    GridLabel::Dense | GridLabel::Transitional => (),
                    _ => return false,
                };
            }
        }
        true
    }

    fn is_inside_grid(dg1: DG, other_dgs: Vec<DG>) -> bool {
        let dgs_wo_1 = other_dgs.into_iter().filter(|dg| !(dg.i == dg1.i && dg.j == dg1.j));
        let mut has_neighbor_i = false;
        let mut has_neighbor_j = false;
        for dg in dgs_wo_1 {
            if TheWorld::are_neighbors_in_i(&dg1, &dg) { has_neighbor_i = true; }
            if TheWorld::are_neighbors_in_j(&dg1, &dg) { has_neighbor_j = true; }
        }
        if has_neighbor_i && has_neighbor_j {
            true
        } else {
            false
        }
    }

    fn is_a_grid_group(dgs: Vec<DG>) -> bool {
        let neighbors_graph = TheWorld::dgs_to_graph(dgs.clone());
        let edge_count = neighbors_graph.edge_count();
        let node_count = neighbors_graph.node_count();

        TheWorld::graph_is_fully_connected(neighbors_graph)
    }

    fn are_neighbors(dg1: &DG, dg2: &DG) -> bool {
        TheWorld::are_neighbors_in_i(dg1, dg2) || TheWorld::are_neighbors_in_j(dg1, dg2)
    }

    fn are_neighbors_in_i(dg1: &DG, dg2: &DG) -> bool {
        (dg1.i == dg2.i) && (dg1.j as i32 - dg2.j as i32).abs() == 1
    }

    fn are_neighbors_in_j(dg1: &DG, dg2: &DG) -> bool {
        (dg1.j == dg2.j) && (dg1.i as i32 - dg2.i as i32).abs() == 1
    }

    fn compute_grid_indxs(val: (f64, f64), i_range: (f64, f64), j_range: (f64, f64), i_bins: usize, j_bins: usize) -> Result<(usize, usize), String> {
        let i_size = (i_range.1 - i_range.0) / i_bins as f64;
        let i_number_of_sizes = ((val.0 - i_range.0) / i_size) as usize;
        let j_size = (j_range.1 - j_range.0) / j_bins as f64;
        let j_number_of_sizes = ((val.1 - j_range.0) / j_size) as usize;

        assert!(i_number_of_sizes < i_bins);
        assert!(j_number_of_sizes < j_bins);

        Ok((i_number_of_sizes, j_number_of_sizes))
    }

    pub fn init(&mut self, def_bucket: Vec<GridPoint>) {
        let props: DStreamProps = DStreamProps { ..Default::default() };

        for i in 0..props.i_bins {
            for j in 0..props.j_bins {
                let z_clone = def_bucket.clone();
                let some_default_dg = DG {i: i, j: j,
                    updates_and_vals: z_clone, removed_as_spore_adic: Vec::new(),};
                (self.g_vec).push(((i as usize, j as usize), some_default_dg));
            }
        }
    }
    fn do_time_steps() {}
    fn do_one_time_step(t: u32, data: Vec<RawData>) {}

    fn get_by_idx(&mut self, idx: (usize, usize))-> DG {
        let  vec_f: Vec<((usize, usize), DG)> = self.g_vec.clone().into_iter().filter(|i| i.0 == idx).collect();
        vec_f[0].1.clone()
    }
    fn update_by_idx(&mut self, idx: (usize, usize), dg: DG) -> Result<(), String> {
        self.g_vec.retain(|i| i.0 != idx);
        self.g_vec.push((idx, dg));
        Ok(())
    }
    pub fn put(&mut self, t: u32, dat: Vec<RawData>) -> Result<(), String> {

        self.current_time = t;
        if ! self.timeline.contains(&t) {
            self.timeline.push(t);
        } else {
            return Err(String::from("time has already been put"));
        }

        fn validate_range(loc2d: (f64, f64)) -> bool {
            let props: DStreamProps = DStreamProps { ..Default::default() };
            if (loc2d.0 <= props.i_range.1) &&
               (loc2d.0 >= props.i_range.0) &&
               (loc2d.1 <= props.j_range.1) &&
               (loc2d.1 >= props.j_range.0) {
            } else {
                println!("invalid! -- bad input range");
                return false
            }
            true
        }

        let with_idxs: Vec<GridData> = dat
            .iter()
            .filter(|rd| validate_range((rd.x, rd.y)))
            .map(|rd| self.which_idxs(rd).unwrap())
            .collect();

        let props: DStreamProps = DStreamProps { ..Default::default() };

        for (key, group) in with_idxs.iter().group_by(|gd| (gd.i, gd.j)) {
            let the_vec_of_vals: Vec<f64> = group.iter().map(|t| t.v).collect();
            let mut some_default_dg = DG {i: key.0, j: key.1, updates_and_vals: Vec::<GridPoint>::new(), removed_as_spore_adic: Vec::new(),};
            let teh_dg: &mut DG = &mut self.get_by_idx(key);
            let update_dg_result = teh_dg.update(t, the_vec_of_vals);
            let udpate_world_result = self.update_by_idx(key, teh_dg.clone());
        }
        Ok(())
    }
    fn which_idxs(&self, dat: &RawData) -> Result<GridData, String> {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        let idxs = TheWorld::compute_grid_indxs((dat.x, dat.y), props.i_range, props.j_range,props.i_bins, props.j_bins).unwrap();
        Ok((GridData{i:idxs.0,j:idxs.1,v:dat.v}))
    }

    fn get_neighbors(the_dg: DG, other_dgs: Vec<DG>) -> Vec<DG> {
        other_dgs.clone().into_iter().filter( |dg| TheWorld::are_neighbors(&dg, &the_dg) ).collect()
    }
}

impl DG {

    pub fn is_sporadic (&self, t: u32) -> bool {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        let last_update_t_and_v = self.get_last_update_and_value_to(t);
        let d_t = self.get_at_time(t).1;
        let n_size = (self.i * self.j) as f64;
        let pi = (props.c_l * (1.0 - props.lambda.powf((t - last_update_t_and_v.0 + 1) as f64))) / (n_size * (1.0 - props.lambda));
        let label = self.get_grid_label_at_time(t);
        match label {
            GridLabel::Sparse => {
                let last_t_removed_spore = self.get_last_time_removed_as_sporadic_to(t);
                //TODO if has not been deleted before this should be false, but right now it blows up
                d_t < pi && t as f64 >= (1.0 + props.beta) * (last_t_removed_spore as f64)
            },
            _ => false
        }
    }
    pub fn get_grid_label_at_time(&self, t:u32) -> GridLabel {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        let n_size = (self.i * self.j) as f64;
        let d_m = props.c_m / (n_size * (1.0 - props.lambda));
        let d_l = props.c_l / (n_size * (1.0 - props.lambda));
        let d_t = self.get_at_time(t).1;
        if d_t >= d_m {
            GridLabel::Dense
        } else if d_t <= d_l {
            GridLabel::Sparse
        } else {
            assert!(d_t > d_l && d_t < d_t);
            GridLabel::Transitional
        }
    }
    pub fn get_at_time(&self, t: u32) -> (u32, f64) {
        let last_update_time_and_value = self.get_last_update_and_value_to(t);
        let coeff = self.coeff(t, last_update_time_and_value.0);
        (last_update_time_and_value.0, coeff * last_update_time_and_value.1 + 1.0)
    }
    pub fn update(&mut self, t: u32, vals: Vec<f64>) -> Result<(), String> {
        let sum = vals.clone().iter().fold(0.0, |sum, x| sum + x);
        self.updates_and_vals.push(GridPoint {t: t, v: sum});
        Ok(())
    }
    pub fn get_last_update_and_value_to(&self, t: u32) -> (u32, f64) {
        let a: GridPoint = self.updates_and_vals.clone().into_iter().filter(|bp| bp.t < t).max_by_key(|bp| bp.t).unwrap();
        let t_l = a.t;
        let v_l = a.v;
        (t_l, v_l)
    }
    pub fn get_last_time_removed_as_sporadic_to(&self, t: u32) -> u32 {
        let t = self.removed_as_spore_adic.clone().into_iter().filter(|&the_t| the_t < t).max().unwrap();
        t
    }
    fn coeff(&self, t_n: u32, t_l: u32) -> f64 {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        props.lambda.powf((t_n - t_l) as f64)
    }
 }

pub fn adjust_clustering(grid_list: Vec<RawData>) -> Result<(), String> {
    //update the density of all grids in grid_list
    /*
    foreach grid g whose attribute (dense/sparse/transitional) is changed since last call to adjust_clustering()
        if g is a sparse grid
            delete g from its cluster c, label g as NO_CLASS
            if (c becomes unconnected) split c into two clusters
        else if g is a dense grid
            among all neighboring grids of g, find out the grid h whoe cluster c_h has the largest size
            if h is a dense grid
                if (g is labelled as NO_CLASS) label g as in c_h
                else if (g is in cluster c and |c| > |c_h|)
                    label all grids in c_h as in c
                else if (g is in cluster c and |c| <= |c_h|)
                    label all grids in c as in c_h
            else if h is a transitional grid
                if ((g is NO_CLASS) and (h is an outside grid if g is added to c_h)) label g as in c_h
                else if (g is in cluster c and |c| >= |c_h|)
                    move h from cluster c_h to c
        else if (g is transitional grid)
            among neighboring clusters of g, find the largest one c'
            satisfying that g is an outside grid if added to it and
            label g as in c'
    */

    Ok(())
}
