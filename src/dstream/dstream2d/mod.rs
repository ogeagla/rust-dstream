use na::{Mat2, DMat};
use std::collections::{HashMap, HashSet};
use std::num::*;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;
use rand;

mod test;

// TODO horrifying use of clone in this whole file...

// TODO learnings from work on the impl of init clusters:
// - NO_CLASS refers to a grid no being in a cluster. so the grid's 'cluster key' is 'NO_CLASS'
// - grids need to know their status, label, and which cluster they belong to at a time. period!

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster<'a> {

    //TODO FIXME a Vec of refs doesnt seem right?
    dgs: Vec<&'a GridCell<'a>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridCell<'a> {
    i: usize,
    j: usize,
    updates_and_vals: Vec<GridPoint>,
    removed_as_spore_adic: Vec<u32>,
    cluster: Option<&'a Cluster<'a>>,
    label: GridLabel,
    status: GridStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GridLabel { Dense, Sparse, Transitional, }

#[derive(Debug, PartialEq, Clone)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
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
pub struct TheWorld<'a> {
    clusters: Vec<Cluster<'a>>,
    grid_cells: Vec<GridCell<'a>>,
    previous_grid_cells: Vec<GridCell<'a>>,
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
        let mut world = TheWorld{grid_cells: Vec::new(), timeline: Vec::new(), 
            current_time: 0, clusters: Vec::new(), previous_grid_cells: Vec::new(), };
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
                    println!("-- adjusting clusters");
                    let result = world.adjust_clustering();
                } else {
                    println!("-- initializing clusters");
                    let result = world.initialize_clustering();
                    has_initialized = true;
                }
            }
        }
    }
}

impl<'a> TheWorld<'a> {

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

    fn is_outside_when_added_to(dg_to_check_if_outside: GridCell, dg_to_add: GridCell, dgs: Vec<GridCell>) -> bool {
        let mut added_vec = dgs.clone();
        added_vec.push(dg_to_add);
        if TheWorld::is_inside_grid(dg_to_check_if_outside, added_vec) { false } else { true }
    }

    fn get_labels_for_time(t: u32, dgs: Vec<GridCell>) -> HashMap<(usize, usize), GridLabel> {
        let mut the_map = HashMap::new();
        dgs.into_iter().map(|dg| {
            the_map.insert((dg.i, dg.j), dg.get_grid_label_at_time(t));
        });
        the_map
    }

    fn labels_changed_between(labels1: HashMap<(usize, usize), GridLabel>, labels2: HashMap<(usize, usize), GridLabel>) -> bool {
        if labels1.len() != labels2.len() { return true }
        for (idxs, label) in labels1 {
            match labels2.get(&idxs) {
                None => {
                    return true
                },
                Some(l) => if label != *l {
                    return true
                },
            }
        }
        false
    }

    fn which_labels_changed_between(labels1: HashMap<(usize, usize), GridLabel>, labels2: HashMap<(usize, usize), GridLabel>) -> Option<Vec<(usize, usize)>> {

        let mut keys_set1 = HashSet::new();
        for (k, v) in labels1.clone() {
            keys_set1.insert(k);
        }

        let mut keys_set2 = HashSet::new();
        for (k, v) in labels2.clone() {
            keys_set2.insert(k);
        }

        let keys_intersection: HashSet<_> = keys_set1.intersection(&keys_set2).collect();
        let keys_union: HashSet<_> = keys_set1.union(&keys_set2).collect();

        if labels1.len() == labels2.len() && keys_intersection.len() == labels1.len() {
            //keys are identical
            let mut the_changed: Vec<(usize, usize)> = Vec::new();
            for (k, v) in labels1.clone() {
                let val2 = labels2.get(&k).unwrap();
                if *val2 != v {
                    the_changed.push(k);
                }
            }
            if the_changed.len() != 0 { return Some(the_changed) }
        } else {
            let diff_keys: HashSet<_> = keys_union.symmetric_difference(&keys_intersection).collect();
            let mut the_changed: Vec<(usize, usize)> = Vec::new();
            for k in keys_intersection.clone().into_iter() {
                let val1 = labels1.get(&k).unwrap();
                let val2 = labels2.get(&k).unwrap();
                if *val1 != *val2 {
                    the_changed.push(*k);
                }
            }
            for k in diff_keys {
                //TODO this double deref makes me feel uneasy
                the_changed.push(**k);
            }
            return Some(the_changed);
        }

        None
    }

    pub fn adjust_clustering(&mut self) -> Result<(), String> {

        /*
        pub struct TheWorld<'a> {
            clusters: Vec<Cluster<'a>>,
            grid_cells: Vec<GridCell<'a>>,
            timeline: Vec<u32>,
            current_time: u32,
        }
        */

        fn dgs_with_changed_labels_since_last_time<'b>() -> Vec<GridCell<'b>> {
            //TODO
            Vec::new()
        }

        fn is_no_longer_fully_connected(c: Cluster) -> bool {
            //TODO
            //FIXME this code looks horrific...
            let neighbors_graph = TheWorld::dgs_to_graph(
                c
                    .dgs
                    .clone()
                    .iter()
                    .map(|p|
                        (*p).clone())
                    .collect()
            );
            TheWorld::graph_is_fully_connected(neighbors_graph)
        }

        fn get_neighboring_dg_with_largest_cluster(ref_dg: GridCell) -> GridCell {
            //TODO
            GridCell {i: 1, j: 1, updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal}
        }

        for g in dgs_with_changed_labels_since_last_time() {
            match g.get_grid_label_at_time(self.current_time) {
                GridLabel::Sparse => {
                    //TODO
                    //remove g from its cluster
                    //label g as NoClass
                    let c = Cluster { dgs: Vec::new(), };

                    if is_no_longer_fully_connected(c) {
                        //split c into 2 clusters
                    }
                },
                GridLabel::Dense => {
                    let h = get_neighboring_dg_with_largest_cluster(g.clone());
                    match h.get_grid_label_at_time(self.current_time) {
                        GridLabel::Dense => {

                        },
                        GridLabel::Transitional => {

                        },
                        _ => (),
                    }
                },
                _ => (),
            }
        }

        //update the density of all grids in grid_list
        /*
        foreach grid g whose attribute (dense/sparse/transitional) is changed since last call to adjust_clustering()
            if g is a sparse grid
                delete g from its cluster c, label g as NO_CLASS
                if (c becomes unconnected) split c into two clusters
            else if g is a dense grid
                among all neighboring grids of g, find out the grid h whose cluster c_h has the largest size
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

    pub fn initialize_clustering(&mut self) -> Result<(), String> {
        fn single_init_iteration(cs: Vec<Vec<((usize, usize), GridCell)>>, t: u32) -> Vec<Vec<((usize, usize), GridCell)>> {

            let all_g: Vec<((usize, usize), GridCell)> = cs.clone().into_iter().fold(Vec::new(), |mut acc, x| { acc.extend(x); acc });

            for c in cs.clone() {

                for g in c.clone() {

                    if TheWorld::is_inside_grid(g.clone().1, c.clone().into_iter().map(|dg| dg.1).collect()) {
                        //not outside grid
                    } else {
                        //outside grid
                        for h in all_g.clone() {
                            if TheWorld::are_neighbors(&(h.clone().1), &(g.clone().1)) {
                                //for every neighboring grid h
                                for c_prime in cs.clone() {
                                    if c_prime.contains(&h) {
                                        //where h belongs to cluster c_prime
                                        if c_prime.len() > c.len() {
                                            //move all grids in c_prime to c
                                        } else {
                                            //move all grids in c to c_prime
                                        }
                                    }
                                }
                                match h.1.get_grid_label_at_time(t) {
                                    GridLabel::Transitional => {
                                        //move h into c
                                    },
                                    _ => (),
                                }
                            }
                        }
                    }
                }
            }
            Vec::new()
        }

        let mut clusters: Vec<Vec<GridCell>> = self.grid_cells.clone().into_iter().map(|dg| vec!(dg)).collect();

        let all_g: Vec<GridCell> = clusters.clone().into_iter().fold(Vec::new(), |mut acc, x| { acc.extend(x); acc });

        let mut init_labels = TheWorld::get_labels_for_time(self.current_time, all_g);

        while TheWorld::labels_changed_between(init_labels.clone(), init_labels.clone()) {

        }

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

    fn dgs_to_graph(dgs: Vec<GridCell>) ->  Graph<(usize, usize), (usize, usize)> {
        let mut neighbors_graph = Graph::<(usize, usize), (usize, usize)>::new();

        let dgs_and_nidxs: Vec<(GridCell, NodeIndex)> = dgs.clone().into_iter().map(|dg| (dg.clone(), neighbors_graph.add_node((dg.i, dg.j)))).collect();

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

    fn mark_and_remove_and_reset_spore_adics(t: u32, dgs: Vec<GridCell>) {
        //prop 4.3: mark as sporadic; grids marked sporadic last t can be removed this t, or labeled as normal
        //TODO
    }

    ///  if every inside grid of G is a dense grid and
    ///  every outside grid is either a dense grid or a transitional
    ///  grid, then G is a grid cluster.
    fn is_a_grid_cluster(t: u32, dgs: Vec<GridCell>) -> bool {
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

    fn is_inside_grid(dg1: GridCell, other_dgs: Vec<GridCell>) -> bool {
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

    fn is_a_grid_group(dgs: Vec<GridCell>) -> bool {
        let neighbors_graph = TheWorld::dgs_to_graph(dgs.clone());
        let edge_count = neighbors_graph.edge_count();
        let node_count = neighbors_graph.node_count();

        TheWorld::graph_is_fully_connected(neighbors_graph)
    }

    fn are_neighbors(dg1: &GridCell, dg2: &GridCell) -> bool {
        TheWorld::are_neighbors_in_i(dg1, dg2) || TheWorld::are_neighbors_in_j(dg1, dg2)
    }

    fn are_neighbors_in_i(dg1: &GridCell, dg2: &GridCell) -> bool {
        (dg1.i == dg2.i) && (dg1.j as i32 - dg2.j as i32).abs() == 1
    }

    fn are_neighbors_in_j(dg1: &GridCell, dg2: &GridCell) -> bool {
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
                let some_default_dg = GridCell {i: i, j: j,
                    updates_and_vals: z_clone, removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
                (self.grid_cells).push(some_default_dg);
            }
        }
    }
    fn do_time_steps() {}
    fn do_one_time_step(t: u32, data: Vec<RawData>) {}

    fn get_by_idx(&mut self, idx: (usize, usize))-> GridCell<'a> {
        let  vec_f: Vec<GridCell> = self.grid_cells.clone().into_iter().filter(|i| (i.i, i.j) == idx).collect();
        vec_f[0].clone()
    }
    fn update_by_idx(&mut self, idx: (usize, usize), dg: GridCell<'a>) -> Result<(), String> {
        self.grid_cells.retain(|i| (i.i, i.j) != idx);
        self.grid_cells.push(dg);
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
            let mut some_default_dg = GridCell {i: key.0, j: key.1, updates_and_vals: Vec::<GridPoint>::new(), removed_as_spore_adic: Vec::new(), cluster: None, label: GridLabel::Sparse, status: GridStatus::Normal};
            let teh_dg: &mut GridCell = &mut self.get_by_idx(key);
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

    fn get_neighbors(the_dg: GridCell<'a>, other_dgs: Vec<GridCell<'a>>) -> Vec<GridCell<'a>> {
        other_dgs.clone().into_iter().filter( |dg| TheWorld::are_neighbors(&dg, &the_dg) ).collect()
    }
}

impl<'a> GridCell<'a> {

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
