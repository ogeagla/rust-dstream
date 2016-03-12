use na::{Mat2, DMat};
//use na;
use std::num::*;
mod test;

#[test]
fn test_which_grid() {
    let loc = (-6., 6.);
    let i_rn = (-10.0, 10.0);
    let j_rn = (-10.0, 10.0);
    let i_bins = 10 as usize;
    let j_bins = 10 as usize;

    let result = grid_helpers::which_grid(loc, i_rn, j_rn, i_bins, j_bins).unwrap();
    assert_eq!((2, 8), result);

}

mod grid_helpers {
    pub fn which_grid(val: (f64, f64), i_range: (f64, f64), j_range: (f64, f64), i_bins: usize, j_bins: usize) -> Result<(usize, usize), String> {
        let i_size = (i_range.1 - i_range.0) / i_bins as f64;
        let i_number_of_sizes = ((val.0 - i_range.0) / i_size) as usize;
        let j_size = (j_range.1 - j_range.0) / j_bins as f64;
        let j_number_of_sizes = ((val.1 - j_range.0) / j_size) as usize;

        assert!(i_number_of_sizes < i_bins);
        assert!(j_number_of_sizes < j_bins);


        Ok((i_number_of_sizes, j_number_of_sizes))
    }
}

pub trait DensityGrid {

    fn get_value_at_time(&self, time: u32, lambda: f64, dats: Vec<((usize,usize),Dat)>) -> Result<f64, String>;

    fn compute_decay_coeff_at_time(&self, t_n: u32, t_l: u32, lambda: f64) -> f64 {
        lambda.powf((t_n - t_l) as f64)
    }
}

pub trait DensityGridSpace {

    fn put(&mut self, loc2d: (f64, f64), val: f64, t: u32) -> Result<(), String>;

}

/*

input data: [(x, y, v)]

///some impl
for some time t in 0..:
  _.put t, [(x, y, v)]


//some other impl
fn put (t, [(x, y, v)]):
  [(i, j, sum_vals)] = [x,y,v].map(i,j,v).groupby(i,j).reduce(_+_)
  foreach:
    g = density_grids[i, j]
    g.update(t, v)
    print g.get()
*/
struct RawData;
struct TheWorld;
impl TheWorld {
    fn do_time_steps() {}
    fn do_one_time_step() {}
    fn put() {}
}
struct DG;
impl DG {
    fn update() {}
    fn get() {}
}



#[derive(Debug)]
#[derive(Copy)]
#[derive(Clone)]
pub struct GridPoint {
    i: usize, //location
    j: usize,
    density: f64,
    data_time: u32,
}

#[derive(Clone)]
pub struct Dat {
    x: f64,
    y: f64,
    val: f64,
    t: u32,
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

pub struct TheGrid {
    mat: DMat<GridPoint>,
    props: DStreamProps,
    dats: Vec<((usize, usize), Dat)>,
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

impl DensityGrid for GridPoint {

    fn get_value_at_time(&self, time: u32, lambda: f64, dats: Vec<((usize,usize),Dat)>) -> Result<f64, String> {

        //TODO time - 1 is wrong, it should be biggest time smaller than current
        let coeff = self.compute_decay_coeff_at_time(time, time - 5, lambda);

        let sum_vals = dats.iter().fold(0.0, |sum, x| sum + x.1.val);
        println!("sum vals: {}", sum_vals);
        let new_d = coeff * sum_vals + 1.0;
        println!("new D: {}", new_d);

        Ok(new_d)
    }

//    fn update_with_values(&mut self, vals: &[f64], current_time: u32, lambda: f64) -> Result<(), String> {
//        //        D(g, t n ) = λ t n −t l D(g, t l ) + 1
//        let coeff = self.compute_decay_coeff_at_time(current_time, self.data_time, lambda);
//        println!("coeff: {}", coeff);
//        let sum_vals = vals.iter().fold(0.0, |sum, x| sum + x);
//        let new_d = coeff * coeff * sum_vals + 1.0;
//        println!("sum vals: {}", sum_vals);
//        println!("new D: {}", new_d);
//        self.density = new_d;
//        self.data_time = current_time;
//        Ok(())
//    }

}

impl DensityGridSpace for TheGrid {

    fn put(&mut self, loc2d: (f64, f64), val: f64, t: u32) -> Result<(), String> {

//        println!("{}", self.density);
//        let props: DStreamProps = DStreamProps { ..Default::default() };
//        let idxs = grid_helpers::which_grid(val, props.i_range, props.j_range, props.i_bins, props.j_bins);
//        let res_update = self.update_grid_pt_density(val, 0, props.lambda);
//        println!("{}", self.density);


        if ((loc2d.0 <= self.props.i_range.1) &&
            (loc2d.0 >= self.props.i_range.0) &&
            (loc2d.1 <= self.props.j_range.1) &&
            (loc2d.1 >= self.props.j_range.0)) {

            println!("valid!");
            let idxs =
                grid_helpers::which_grid(loc2d, self.props.i_range, self.props.j_range,
                                         self.props.i_bins, self.props.j_bins).unwrap();
            let gp = &mut self.mat[idxs];

//            gp.update_with_values(&[100.0], t, self.props.lambda);

            self.dats.push((idxs, Dat {x: loc2d.0, y: loc2d.1, val: val, t:t}));

            println!("value at next time: {}",
                gp.get_value_at_time(
                    t+10,
                    self.props.lambda,
                        self.dats
                            .clone()
                            .into_iter()
                            .filter(|i| i.0 == idxs)
                            .collect::<Vec<((usize, usize), Dat)>>()).unwrap());

            println!("dats len: {}", self.dats.len());

            Ok(())
        } else {
            println!("invalid");
            Err(String::from("Invalid range"))
        }
    }
}
#[test]
fn test_put() {

    let mut grid_pt_1: GridPoint = GridPoint {i: 0, j: 0,
        density: 0.0, data_time: 0,};

    let mut grid_pt_2: GridPoint = GridPoint {i: 1, j: 0,
        density: 0.0, data_time: 0, };

    let mut grid_pt_3: GridPoint = GridPoint {i: 0, j: 1,
        density: 0.0, data_time: 0, };

    let mut grid_pt_4: GridPoint = GridPoint {i: 1, j: 1,
        density: 0.0, data_time: 0,};

    let mut v = Vec::new();
    for i in 0..100 {
        v.push(grid_pt_1)
    }

    let mut mat_thing = DMat::from_col_vec(
        10,
        10,
        &v
    );

    for i in 0..10 {

        for j in 0..10 {

            let mut g1= GridPoint
            {i: i, j: j,
                density: 0.0, data_time: 0, };

            mat_thing[(i,j)] = g1;
        }

    }

    let mut the_grid = TheGrid {mat: mat_thing, props: DStreamProps { ..Default::default() }, dats: Vec::new()};

    let res_put = the_grid.put((-6.0, 6.0), 100.0, 1).unwrap();
    let res_put = the_grid.put((-6.0, 6.0), 100.0, 10).unwrap();
    let res_put = the_grid.put((-6.0, 6.0), 100.0, 100).unwrap();
    let res_put = the_grid.put((-6.0, 6.0), 100.0, 1000).unwrap();
    let res_put = the_grid.put((-6.0, 6.0), 100.0, 10000).unwrap();


    for i in 0..10 {
        for j in 0..10 {
            let gp: &GridPoint = &the_grid.mat[(i as usize, j as usize)];
            assert_eq!(0.0, gp.density)
        }
    }
}

#[test]
fn test_initialize_clustering() {
    const loc_i: usize = 0 as usize;
    const loc_j: usize = 0 as usize;
    const G1: GridPoint = GridPoint
    {i: loc_i, j: loc_j,
        density: 0.0, data_time: 0, };
    let result = initialize_clustering(DMat::from_row_vec(2,2, &[G1, G1, G1, G1]));
    assert_eq!(Ok(()), result);
}

pub fn initialize_clustering(grid_list: DMat<GridPoint>) -> Result<(), String> {
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


pub fn adjust_clustering(grid_list: DMat<GridPoint>) -> Result<(), String> {
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