use na::{Mat2, DMat};
use std::collections::HashMap;
use std::num::*;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;
use petgraph::{Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::*;

mod test;

#[derive(Debug)]
#[derive(Clone)]
struct DG {
    i: usize,
    j: usize,
    updates_and_vals: Vec<GridPoint>,
    removed_as_spore_adic: Vec<u32>,
}

#[derive(Debug)]
#[derive(Clone)]
enum GridLabel { Dense, Sparse, Transitional, }

#[derive(Debug)]
#[derive(Clone)]
enum GridStatus { Sporadic, Normal, }

pub struct DStreamProps {
    c_m: f64,
    c_l: f64,
    lambda: f64,
    beta: f64,
    i_bins: usize,
    j_bins: usize,
    i_range: (f64, f64),
    j_range: (f64, f64),
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
struct GridPoint {
    t: u32,
    v: f64,
}
#[derive(Clone)]
struct GridData {
    i: usize,
    j: usize,
    v: f64,
}
struct TheWorld {
    g_vec: Vec<((usize, usize), DG)>,
}

impl Default for DStreamProps {
    fn default() -> DStreamProps {
        DStreamProps {
            c_m: 3.0,
            c_l: 0.8,
            lambda: 0.998,
            beta: 0.3,
            i_bins: 10 as usize,
            j_bins: 10 as usize,
            i_range: (-10.0, 10.0),
            j_range: (-10.0, 10.0),
        }
    }
}

impl TheWorld {

    fn mark_and_remove_and_reset_spore_adics(t: u32, dgs: Vec<DG>) {
        //prop 4.3: mark as sporadic; grids marked sporadic last t can be removed this t, or labeled as normal
        //TODO
    }

    fn is_a_grid_cluster(dg1: DG, other_dgs: Vec<DG>) -> bool {
        //TODO
        false
    }

    fn is_inside_grid(dg1: DG, other_dgs: Vec<DG>) -> bool {
        //TODO
        false
    }

    fn is_a_grid_group(dgs: Vec<DG>) -> bool {
        //TODO
        //def 3.4 of paper

        let mut neighbors_graph = Graph::<(usize, usize), (usize, usize)>::new();

        let mut nodes: Vec<NodeIndex> = Vec::new();

        for dg in dgs.clone().into_iter() {
            let idx = neighbors_graph.add_node((dg.i, dg.j));
            nodes.push(idx);
        }

        for the_dgs in dgs.clone().into_iter().combinations_n(2) {

            let i1 = the_dgs[0].i;
            let j1 = the_dgs[0].j;
            let i2 = the_dgs[1].i;
            let j2 = the_dgs[1].j;

            let r_neighbors = TheWorld::are_neighbors(&the_dgs[0], &the_dgs[1]);
            println!("checking if pair are neighbors: {} {} and {} {} => {}", i1, j1, i2, j2, r_neighbors);

            let node1 = neighbors_graph.add_node((i1, j1));
            let node2 = neighbors_graph.add_node((i2, j2));

            if r_neighbors {
                neighbors_graph.extend_with_edges(&[(node1, node2),]);
            }
        }

        let edge_count = neighbors_graph.edge_count();
        let node_count = neighbors_graph.node_count();

        println!("edge count: {}", edge_count);
        println!("node count: {}", node_count);


        for n in nodes.into_iter() {
            let neighbors = neighbors_graph.neighbors_undirected(n);
            let size_hint = neighbors.size_hint();
            for neighbor in neighbors {
                println!("neighbor...");
            }
            println!("  neighbors size: {}", size_hint.0);
            match size_hint.1 {
              Some(s) => println!(" more size hint: {}", s),
              None => (),
            };
        }

        let is_it = (edge_count + 1) >= dgs.len();
        println!("-> {}", is_it);
        is_it
    }


    fn are_neighbors(dg1: &DG, dg2: &DG) -> bool {
        TheWorld::are_neighbors_in_i(dg1, dg2) || TheWorld::are_neighbors_in_j(dg1, dg2)
    }

    fn are_neighbors_in_i(dg1: &DG, dg2: &DG) -> bool {
        (dg1.i == dg2.i) && (dg1.j as i32 - dg2.j as i32).abs() <= 1
    }

    fn are_neighbors_in_j(dg1: &DG, dg2: &DG) -> bool {
        (dg1.j == dg2.j) && (dg1.i as i32 - dg2.i as i32).abs() <= 1
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

    fn init(&mut self, def_bucket: Vec<GridPoint>) {
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
    fn put(&mut self, t: u32, dat: Vec<RawData>) -> Result<(), String> {

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
}

impl DG {

    fn is_sporadic (&self, t: u32) -> bool {
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
    fn get_grid_label_at_time(&self, t:u32) -> GridLabel {
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
    fn get_at_time(&self, t: u32) -> (u32, f64) {
        let last_update_time_and_value = self.get_last_update_and_value_to(t);
        let coeff = self.coeff(t, last_update_time_and_value.0);
        (last_update_time_and_value.0, coeff * last_update_time_and_value.1 + 1.0)
    }
    fn update(&mut self, t: u32, vals: Vec<f64>) -> Result<(), String> {
        let sum = vals.clone().iter().fold(0.0, |sum, x| sum + x);
        self.updates_and_vals.push(GridPoint {t: t, v: sum});
        Ok(())
    }
    fn get_last_update_and_value_to(&self, t: u32) -> (u32, f64) {
        let a: GridPoint = self.updates_and_vals.clone().into_iter().filter(|bp| bp.t < t).max_by_key(|bp| bp.t).unwrap();
        let t_l = a.t;
        let v_l = a.v;
        (t_l, v_l)
    }
    fn get_last_time_removed_as_sporadic_to(&self, t: u32) -> u32 {
        let t = self.removed_as_spore_adic.clone().into_iter().filter(|&the_t| the_t < t).max().unwrap();
        t
    }
    fn coeff(&self, t_n: u32, t_l: u32) -> f64 {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        props.lambda.powf((t_n - t_l) as f64)
    }
 }

pub fn initialize_clustering(grid_list: Vec<RawData>) -> Result<(), String> {
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