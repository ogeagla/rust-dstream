use na::{Mat2, DMat};
use std::collections::HashMap;
use std::num::*;
use itertools::Itertools;
use std::cell::RefCell;
use std::rc::Rc;

mod test;

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

#[test]
fn test_new_put_works() {
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

impl TheWorld {

    fn are_neighbors(dg1: &DG, dg2: &DG) -> bool {
        if (dg1.i == dg2.i) && (dg1.j as i32 - dg2.j as i32).abs() <= 1 {
            return true
        } else if (dg1.j == dg2.j) && (dg1.i as i32 - dg2.i as i32).abs() <= 1 {
            return true
        }
        false
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
        println!("init");
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

            if ((loc2d.0 <= props.i_range.1) &&
            (loc2d.0 >= props.i_range.0) &&
            (loc2d.1 <= props.j_range.1) &&
            (loc2d.1 >= props.j_range.0)) {
                println!("valid!");
            } else {
                println!("invalid! -- bad input range");
                return false
            }
            return true
        }
        let with_idxs: Vec<GridData> = dat
            .iter()
            .filter(|rd| validate_range((rd.x, rd.y)))
            .map(|rd| self.which_idxs(rd).unwrap())
            .collect();

        let props: DStreamProps = DStreamProps { ..Default::default() };


        for (key, group) in with_idxs.iter().group_by(|gd| (gd.i, gd.j)) {
            println!("--put: key, g size: ({},{}) : {}", key.0, key.1, group.len());

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

#[test]
fn test_dg_update_and_get() {
    let mut dg = DG {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: Vec::new(),};
    dg.update(1, vec!(100.0));

    dg.update(10, vec!(200.0));
    dg.update(20, vec!(300.0));

    assert_eq!((1, 100.0), dg.get_last_update_and_value_to(2));
    assert_eq!((10, 200.0), dg.get_last_update_and_value_to(14));
    assert_eq!((20, 300.0), dg.get_last_update_and_value_to(22));

    let v_at_t = dg.get_at_time(30);
    println!("v: {}", v_at_t);
}

#[test]
fn test_removed_as_sporadic() {
    let mut dg = DG {i: 0, j:0, updates_and_vals: Vec::new(), removed_as_spore_adic: vec!(1, 2, 3, 4),};
    assert_eq!(1, dg.get_last_time_removed_as_sporadic_to(2));
    assert_eq!(4, dg.get_last_time_removed_as_sporadic_to(6));
}

#[derive(Debug)]
#[derive(Clone)]
//#[derive(Copy)] TODO: how to make this work?
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
impl DG {

    fn get_grid_label_at_time(&self, t:u32) -> GridLabel {
        let props: DStreamProps = DStreamProps { ..Default::default() };
        let n_size = (self.i * self.j) as f64;
        let d_m = props.c_m / (n_size * (1.0 - props.lambda));
        let d_l = props.c_l / (n_size * (1.0 - props.lambda));

        let d_t = self.get_at_time(t);

        if d_t >= d_m {
            GridLabel::Dense
        } else if d_t <= d_l {
            GridLabel::Sparse
        } else {
            assert!(d_t > d_l && d_t < d_t);
            GridLabel::Transitional
        }
    }

    fn get_at_time(&self, t: u32) -> f64 {
        let last_update_time_and_value = self.get_last_update_and_value_to(t);
        let coeff = self.coeff(t, last_update_time_and_value.0);
        coeff * last_update_time_and_value.1 + 1.0
    }
    fn update(&mut self, t: u32, vals: Vec<f64>) -> Result<(), String> {
        let sum = vals.clone().iter().fold(0.0, |sum, x| sum + x);
        self.updates_and_vals.push(GridPoint {t: t, v: sum});
        println!("  times updated dg: {}", self.updates_and_vals.len());
        Ok(())
    }

    fn get_last_update_and_value_to(&self, t: u32) -> (u32, f64) {
        let a: GridPoint = self.updates_and_vals.clone().into_iter().filter(|bp| bp.t < t).max_by_key(|bp| bp.t).unwrap();
        let t_l = a.t;
        let v_l = a.v;
        println!("  last update relative to {} is (t: {}, v: {})", t, t_l, v_l);
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

#[test]
fn test_initialize_clustering() {

    let result = initialize_clustering(Vec::new());
    assert_eq!(Ok(()), result);
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