use na::*;
use std::num::*;
mod test;

#[test]
fn test_initialize_clustering() {
    const loc_i: usize = 0 as usize;
    const loc_j: usize = 0 as usize;
    const G1: GridPoint = GridPoint
    {i: loc_i, j: loc_j,
        density: 0.0, new_data_time: 0, last_data_time: 0,
        last_density: 0.0};
    let result = initialize_clustering(Mat2::new(G1, G1, G1, G1));
    assert_eq!(Ok(()), result);
}

#[test]
fn test_which_grid() {
    let val = (3.1, 1.1);
    let i_rn = (0.0, 10.0);
    let j_rn = (-2.0, 2.0);
    let i_bins = 10 as usize;
    let j_bins = 4 as usize;

    let result = GridHelpers::which_grid(val, i_rn, j_rn, i_bins, j_bins).unwrap();
    assert_eq!(3, result.0);
    assert_eq!(3, result.1);
}

mod GridHelpers {
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

pub trait GridSpace {
    fn update_grid_pt_density(&mut self, val_to_add: (f64, f64), current_time: u32, lambda: f64) -> Result<(), String>;

    fn compute_decay_factor_at_time(&self, t_n: u32, t_l: u32, lambda: f64) -> f64 {
        lambda.powf((t_n - t_l) as f64)
    }

}

pub struct GridPoint {
    i: usize,
    j: usize,
    density: f64,
    last_density: f64,
    new_data_time: u32,
    last_data_time: u32,
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

impl GridSpace for GridPoint {

    //TODO
    fn update_grid_pt_density(&mut self, val_to_add: (f64, f64), current_time: u32, lambda: f64) -> Result<(), String> {
        let coeff = self.compute_decay_factor_at_time(self.new_data_time, self.last_data_time, lambda);
        let new_density = coeff * self.last_density + 1.0;

        self.last_density = self.density;
        self.density = new_density;

        self.last_data_time = self.new_data_time;
        self.new_data_time = current_time;

        Ok(())
    }
}

pub fn initialize_clustering(grid_list: Mat2<GridPoint>) -> Result<(), String> {
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


pub fn adjust_clustering(grid_list: Mat2<GridPoint>) -> Result<(), String> {
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